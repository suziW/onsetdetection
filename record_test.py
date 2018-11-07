#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import librosa
from parse_note import Parse_note

class Preprocess:
    def __init__(self, input_dir, window_size=60, step=1/3, sr=22050):   # window_size: ms; step: scale of window_size
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self.__step = math.floor(self.__window_size*step)
        self.__sr = sr

        self.__wavfile = input_dir
        self.__get_aligned_xy()

    def __get_aligned_xy(self):
        self.__x_input = []
        wav_file = self.__wavfile
        wav_name = os.path.split(wav_file)[1]
        # get wav
        wav, _ = librosa.load(wav_file, self.__sr)
        # wav_start_time = int(wav_start_times[wav_name]*sr)
        # wav = wav[wav_start_time:]
        print('>>>>>>>>>>> wav:', wav_file, wav.shape)

        for i in np.arange(0, len(wav)-self.__window_size+1, self.__step):
            self.__x_input.append(wav[i:i+self.__window_size])

        self.__x_input = np.array(self.__x_input)

    def get_wav(self):
        return self.__x_input

class DataGen:
    def __init__(self, wav_file, batch_size=128, window_size=60, step=1/3, sr=22050):
        print('>>>>>>>>>>>>>>>>>> getting data from wav_file: ', wav_file)       
        self.__framepms = int(sr//1000)
        self.__window_size = window_size
        self._batch_size = batch_size
        self.__wav_dir = wav_file
        self.__step = step
        self.__sr = sr

        self.__preprocess()
        self.i = 0

    def getinfo_wav(self):
        return self.__x_inputs.shape, self.wav_steps()

    def wav_steps(self):
        return math.ceil(self.__x_inputs.shape[0]/self._batch_size)

    def wav_gen(self):
        while True:
            if (self.i + 1) * self._batch_size > self.__x_inputs.shape[0]:
                # return rest and then switch files
                x = self.__x_inputs[self.i * self._batch_size:]
                self.i = 0
            else:
                x = self.__x_inputs[self.i * self._batch_size:(self.i + 1) * self._batch_size]
                self.i += 1
            yield x

    def __preprocess(self):
        prep = Preprocess(self.__wav_dir, window_size=self.__window_size, step=self.__step, sr=self.__sr)
        self.__x_inputs = prep.get_wav()

class Predict:
    def __init__(self, input_dir, model_dir, meta_name):
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.meta_name = meta_name
        self.predict()

    def predict(self):
        data = DataGen(self.input_dir, batch_size=256)
        print('>>>>>>>>>>>>>>>>>> data info:', data.getinfo_wav())
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(os.path.join(model_dir, 'savers/', meta_name))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(model_dir, 'savers/')))

            graph = tf.get_default_graph()
            X = graph.get_operation_by_name('x_input').outputs[0]
            Y = graph.get_tensor_by_name('y_input:0')
            training = graph.get_tensor_by_name('training_flag:0')
            keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
            prediction = tf.get_collection('pred_collection')[0]
            # ww = graph.get_tensor_by_name('ww:0')

            self.y_prediction = []
            for i in range(data.wav_steps()):
                batch_x = next(data.wav_gen())
                batch_x = preprocessing.StandardScaler().fit_transform(batch_x.T).T
                batch_y = np.zeros(batch_x.shape[0])
                print('predicting.........................................{}/{}.......'.format(i, data.wav_steps()), end='\r')
                batch_pred = sess.run(prediction, feed_dict={X: batch_x , Y: batch_y , keep_prob: 1, training: False})
                self.y_prediction.append(batch_pred)
                # break
            self.y_prediction = np.concatenate(self.y_prediction)[:, 1]
    
    def get_prediction(self):
        return self.y_prediction
        

class Eval:
    """ prediction and groundtruth should be shape (frame, one_hot_notes)
        discard: the time(ms) threshhole of notes u wanna discard detected frome prediction
        threshhole: the threshhole u convert prediction possibilites to one_hot code
        sr: sampling rate of .wav file
        hop_length: hop_length of CQT 
        onset_tolerance: time(ms) u tolerate the difference between prediction note onset and 
                            groundtruth note onst
        offset_tolerance: same as above discribe
    """
    def __init__(self, model_dir, wav_dir, note_dir, meta_name, discard=50, threshhole=0.5,
                sr=22050,  onset_tolerance=1, offset_tolerance=100):
        self.__threshhole = threshhole
        self.__onset_tolerance = onset_tolerance
        self.sr = sr
        self.wav_dir = wav_dir
        self.model_dir = model_dir
        self.meta_name = meta_name

        self.wav_prediction()
        self.y_pred_prob = self.y_pred.copy()
        self.__prob2onehot()
        self.notes = Parse_note(note_dir, self.x_wav.shape[0]).get_note()
        assert(self.x_wav.shape[0] == self.y_pred.shape[0], self.notes.shape[0])
        self.y_pred_pad = np.pad(self.y_pred.reshape(1, -1), ((87, 0), (0, 0)), 'maximum')
        self.notes = self.notes.T


    def wav_prediction(self):
        wav = Preprocess(self.wav_dir)
        prediction = Predict(self.wav_dir, self.model_dir, self.meta_name)
        self.x_wav = wav.get_wav()[:, 440:880]
        self.y_pred = prediction.get_prediction()

    def __prob2onehot(self):
        self.y_pred[self.y_pred>=self.__threshhole] = 1
        self.y_pred[self.y_pred<self.__threshhole] = 0

    def analysis(self, save_dir):
        wav_name = os.path.splitext(os.path.split(self.wav_dir)[1])[0]
        fig = plt.figure(figsize=(50, 10), dpi=100)
        fig.tight_layout()

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print('making path  ', save_dir)

        wav = self.x_wav.reshape(-1)
        S = librosa.hybrid_cqt(wav, fmin=librosa.midi_to_hz(21), sr=self.sr, hop_length=128,
                                bins_per_octave=4*12,  n_bins=88*4, filter_scale=0.5)


        fig.add_subplot(413)
        plt.pcolormesh(self.y_pred_pad+self.notes*20, cmap='jet')
        fig.add_subplot(414)
        plt.xlim(-0.5, self.x_wav.shape[0]-0.5)
        plt.plot(range(self.x_wav.shape[0]), self.y_pred_prob, 'ro-')   
        fig.add_subplot(411)
        plt.xlim(-0.5, len(wav)-0.5)
        plt.plot(wav)
        fig.add_subplot(412)
        plt.pcolormesh(np.abs(S), cmap='jet')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0, 0)

        plt.savefig('{}/{}.jpg'.format(save_dir,wav_name))
        plt.show()
        plt.clf()
        print('saving presion analysis')


if __name__=='__main__':
    save_dir = 'pic/record_analysis'
    wav_dir = 'data2/record/2018-11-6-test.wav'

    # note_dir = 'txt/wave_test_0.9_0.2_max.txt'
    # note_dir = 'txt/wave_test_0.9_0.51_max.txt'
    note_dir = 'txt/wave_test_0.9_0.6_no_max.txt'
    # note_dir = 'txt/wave_test_0.9_0.7_max.txt'
    # note_dir = 'txt/wave_test_0.9_0.6_no_max_all.txt'
    # note_dir = 'txt/wave_test_0.9_0.8_no_max_all.txt'
    # note_dir = 'txt/wave_test_0.9_0.9_no_max_all.txt'
    # note_dir = 'txt/wave_test_0.9_0.99_no_max_all.txt'

    # model_dir = 'model/mapsfix/'
    # meta_name = '0.9390839993416726-23.0-372485.meta'
    # model_dir = 'model/data1-dense/'
    # meta_name = '0.9265652068026431-18.0-290106.meta'
    # model_dir = 'model/data2-deep/'
    # meta_name = '0.9452492247025172-26.0-403286.meta'
    # model_dir = 'model/data4-deep/'
    # meta_name = '0.9406533798445826-8.0-130976.meta'
    # model_dir = 'model/data5-deep/'
    # meta_name = '0.9586843959120817-8.0-102136.meta'
    model_dir = 'model/deep5-maps-halfstop/'
    meta_name = '0.9607445988082146-6.0-306402.meta'
    # model_dir = 'model/'
    # meta_name = '0.9706520959157352-11.0-561737.meta'

    evaluation = Eval(model_dir, wav_dir, note_dir, meta_name, threshhole=0.8, onset_tolerance=1)
    evaluation.analysis(save_dir)