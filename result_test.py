#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import glob
import tensorflow as tf
import load
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import librosa

def get_predict(input_dir, model_dir, meta_name):
    data = load.DataGen(input_dir, batch_size=256, split=1)
    print('>>>>>>>>>>>>>>>>>> data info:', data.getinfo_train())
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

        y_prediction = []
        y_groundtruth = []
        for i in range(data.train_steps()):
            batch_x , batch_y   = next(data.train_gen())
            batch_x = preprocessing.StandardScaler().fit_transform(batch_x.T).T
            print('predicting.........................................{}/{}.......'.format(i, data.train_steps()), end='\r')
            batch_pred = sess.run(prediction, feed_dict={X: batch_x , Y: batch_y , keep_prob: 1, training: False})
            y_prediction.append(batch_pred)
            y_groundtruth.append(batch_y)
            # break
        y_prediction = np.concatenate(y_prediction)
        y_groundtruth = np.concatenate(y_groundtruth)
        # print('>>>>>>>>>>>>>>>>>> groundtruth: ', y_groundtruth .shape)
        # print('>>>>>>>>>>>>>>>>>> pred: ', y_prediction.shape)
        assert(len(y_groundtruth)==y_prediction.shape[0])

        # ww_data = sess.run(ww)
        # print('>>>>>>>>>>>>>>>>>>>wc1: ', ww_data.shape, type(ww_data))
        # # for i in range(110):
        # plt.figure()
        # plt.plot(ww_data[:, 0])
        # plt.figure()
        # transformed = np.fft.fft(ww_data[:, 0], n=4096)
        # plt.plot(abs(transformed[:]))
        # # axe = [(fft, fft/4096*22050) for fft in range(0, 400, 10)]
        # # print(axe)
        # plt.show()

        print('>>>>>>>>>>>>>>>>>> saving: ', model_dir)
        mm1 = np.memmap(filename=os.path.join(model_dir, 'y_onset.dat'), mode='w+', shape=y_groundtruth .shape[0])
        mm1[:] = y_groundtruth [:]
        mm2 = np.memmap(filename=os.path.join(model_dir, 'y_pred.dat'), mode='w+', shape=y_prediction.shape[0], dtype=float)
        mm2[:] = y_prediction[: ,1]
        del mm1
        del mm2

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
    def __init__(self, model_dir, input_dir, discard=50, threshhole=0.5,
                sr=22050,  onset_tolerance=1, offset_tolerance=100):
        self.__threshhole = threshhole
        self.__onset_tolerance = onset_tolerance
        self.sr = sr
        self.input_dir = input_dir

        mmy_onset = np.memmap(os.path.join(model_dir, 'y_onset.dat'), mode='c')
        self.y_onset = mmy_onset
        mmy_pred = np.memmap(os.path.join(model_dir, 'y_pred.dat'), mode='c', dtype=float)
        self.y_pred = mmy_pred[:]
        self.y_pred_prob = mmy_pred.copy()
        self.__prob2onehot()
        self.y_onset_pad = np.pad(self.y_onset.reshape(1, -1), ((87, 0), (0, 0)), 'maximum')
        self.y_pred_pad = np.pad(self.y_pred.reshape(1, -1), ((87, 0), (0, 0)), 'maximum')

        mmy_groundtruth = np.memmap(input_dir + 'y_groundtruth.dat', mode='c')
        self.y_groundtruth = mmy_groundtruth.reshape(-1, 88).T

        mmx_wav = np.memmap(input_dir + 'x_input.dat', mode='c', dtype=float)
        self.x_wav = mmx_wav.reshape(-1, 1320)[:, 440:880]

        del mmy_onset
        del mmy_pred
        del mmy_groundtruth
        del mmx_wav
        print('>>>>>>>>>>>>>>>>>> shapes: ', self.y_onset.shape, self.y_pred.shape, self.y_groundtruth.shape,
                                         self.y_onset_pad.shape, self.y_pred_pad.shape, self.x_wav.shape)
        assert(self.y_onset.shape == self.y_pred.shape)

    def __prob2onehot(self):
        self.y_pred[self.y_pred>=self.__threshhole] = 1
        self.y_pred[self.y_pred<self.__threshhole] = 0

    def plot(self, begin, end): 
        pred_onset = np.array([self.y_onset, self.y_onset, 2*self.y_pred])
        plt.figure(figsize=(50, 25))
        plt.pcolor(pred_onset[:, begin:end])
        plt.title('pred vs onset')
        # plt.savefig('model/maps 0.77/pic/{}-{}pred vs onset'.format(begin, end))
        
        plt.figure(figsize=(50, 25))
        plt.pcolor(self.y_groundtruth[:, begin:end]+self.y_onset_pad[:, begin:end]*20)
        plt.title('groundtruth onset')
        # plt.savefig('model/maps 0.77/pic/{}-{}truth'.format(begin, end))

        plt.figure(figsize=(50, 25))
        plt.pcolor(self.y_groundtruth[:, begin:end]+self.y_pred_pad[:, begin:end]*20)
        plt.title('groundtruth pred')
        # plt.savefig('model/maps 0.77/pic/{}-{}pred'.format(begin, end))

        plt.show()
        # plt.close()
        if end<len(self.y_onset):
            return True
        return False

    def frameP(self):       # how many predictions are true 
        total = 0
        self.P_false_index = []
        self.P_true_index = []
        for i in range(len(self.y_onset)):
            if self.y_pred[i] == 1:
                total += 1
                if np.sum(self.y_onset[i-self.__onset_tolerance:i+1+self.__onset_tolerance]) > 0:
                    self.P_true_index.append(i) 
                else:
                    self.P_false_index.append(i)
        print('-----------------precision: {}/{}={}'.format(len(self.P_true_index), total, len(self.P_true_index)/total))
        print('-----------------total false is {}'.format(len(self.P_false_index)))
        # print('-----------------scores predict to one, actually zero: ')
        # false_score = [self.y_pred[i] for i in self.P_false_index]
        # print(false_score)
        return len(self.P_true_index)/total

    def frameR(self):       # how many groundtruth are predicted right
        total = 0
        self.R_false_index = []
        self.R_true_index = []
        for i in range(len(self.y_onset)):
            if self.y_onset[i] == 1:
                total += 1
                if np.sum(self.y_pred[i-self.__onset_tolerance:i+1+self.__onset_tolerance]) > 0:
                    self.R_true_index.append(i) 
                else:
                    self.R_false_index.append(i)
        print('-----------------recall: {}/{}={}'.format(len(self.R_true_index), total, len(self.R_true_index)/total))
        print('-----------------total false is {}'.format(len(self.R_false_index)))
        # print('-----------------scores should be one, but predict to zero: ')
        # false_score = [self.y_pred[i] for i in self.R_false_index]
        # print(false_score)    
        return len(self.R_true_index)/total

    def analysis(self):
        print_scalar = 25
        fig_size = 5
        fig = plt.figure(figsize=(1.5*fig_size, 2*fig_size), dpi=100)
        fig.tight_layout()
        onset_pred_split = 44

        for j, i in enumerate(self.P_false_index):
            if i < 26: continue
            if (i+1+print_scalar) > len(self.y_pred): continue
            P_save_dir = 'pic/analysis/{}'.format(self.input_dir.split('/')[-2])
            if not os.path.exists(P_save_dir):
                os.mkdir(P_save_dir)
                os.mkdir(P_save_dir+'/precision')

            hight_light = np.zeros_like(self.y_onset_pad[:, i-print_scalar:i+1+print_scalar])
            hight_light[:, print_scalar] = 100
            hight_light = hight_light[onset_pred_split:, :]
            padding_onset = np.concatenate((40*self.y_onset_pad[:onset_pred_split, i-print_scalar:i+1+print_scalar], hight_light + 20*self.y_pred_pad[onset_pred_split:, i-print_scalar:i+1+print_scalar]),
                                            axis=0)

            wav = self.x_wav[i-print_scalar:i+1+print_scalar, :].reshape(-1)
            S = librosa.hybrid_cqt(wav, fmin=librosa.midi_to_hz(21), sr=self.sr, hop_length=128,
                                    bins_per_octave=4*12,  n_bins=88*4, filter_scale=1)


            fig.add_subplot(413)
            plt.pcolormesh(self.y_groundtruth[:, i-print_scalar:i+1+print_scalar]+padding_onset, vmin=0, vmax=50, cmap='jet')
            fig.add_subplot(414)
            plt.xlim(-0.5, 50.5)
            plt.plot(range(51), self.y_onset[i-print_scalar:i+1+print_scalar], 'ro-') 
            plt.plot(range(51), self.y_pred_prob[i-print_scalar:i+1+print_scalar], 'bo-')   
            fig.add_subplot(411)
            plt.xlim(-0.5, len(wav)-0.5)
            plt.plot(wav)
            fig.add_subplot(412)
            plt.pcolormesh(np.abs(S), cmap='jet')

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0, 0)

            plt.savefig('{}/precision/i-{}__time-{:.2f}__prob-{:.2f}__true.jpg'.format(P_save_dir, i, 440*i/self.sr, self.y_pred_prob[i]))
            # plt.show()
            plt.clf()
            print('saving presion analysis {}/{}'.format(j, len(self.P_false_index)), end='\r')
##########################################################################################################################################################################
##########################################################################################################################################################################
        for j, i in enumerate(self.R_false_index):
            if i < 26: continue
            R_save_dir = 'pic/analysis/{}'.format(self.input_dir.split('/')[-2])
            if not os.path.exists(R_save_dir+'/recall'):
                os.mkdir(R_save_dir+'/recall')

            hight_light = np.zeros_like(self.y_onset_pad[:, i-print_scalar:i+1+print_scalar])
            hight_light[:, print_scalar] = 100
            hight_light = hight_light[onset_pred_split:, :]
            padding_onset = np.concatenate((40*self.y_onset_pad[:onset_pred_split, i-print_scalar:i+1+print_scalar] + hight_light, 20*self.y_pred_pad[onset_pred_split:, i-print_scalar:i+1+print_scalar]),
                                            axis=0)

            wav = self.x_wav[i-print_scalar:i+1+print_scalar, :].reshape(-1)
            S = librosa.hybrid_cqt(wav, fmin=librosa.midi_to_hz(21), sr=self.sr, hop_length=128,
                                    bins_per_octave=4*12,  n_bins=88*4, filter_scale=1)
            
            fig.add_subplot(413)
            plt.pcolormesh(self.y_groundtruth[:, i-print_scalar:i+1+print_scalar]+padding_onset, vmin=0, vmax=50, cmap='jet')
            fig.add_subplot(414)
            plt.xlim(-0.5, 50.5)
            plt.plot(range(51), self.y_onset[i-print_scalar:i+1+print_scalar], 'ro-') 
            plt.plot(range(51), self.y_pred_prob[i-print_scalar:i+1+print_scalar], 'bo-')   
            fig.add_subplot(411)
            plt.xlim(-0.5, len(wav)-0.5)
            plt.plot(wav)
            fig.add_subplot(412)
            plt.pcolormesh(np.abs(S), cmap='jet')

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0, 0)
            
            plt.savefig('{}/recall/i-{}__time-{:.2f}__prob-{:.2f}__pred.jpg'.format(R_save_dir, i, 440*i/self.sr, self.y_pred_prob[i]))
            # plt.show()
            plt.clf()
            print('saving recall analysis {}/{}'.format(j, len(self.R_false_index)), end='\r')


    def frameF(self):
        precision = self.frameP()
        recall = self.frameR()
        print('|||||||||||||||||||f measure is: ', precision*recall*2/(precision+recall))
    def precision(self):
        true_cnt = 0
        for i in range(len(self.y_onset)):
            if self.y_pred[i] == self.y_onset[i]:
                true_cnt += 1
        print('-----------------total precision: ', true_cnt/len(self.y_onset))


if __name__=='__main__':
    input_dir = 'data/maps/test/*/'

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
    # model_dir = 'model/deep5-maps-halfstop/'
    # meta_name = '0.9607445988082146-6.0-306402.meta'
    model_dir = 'model/'
    meta_name = '0.9706520959157352-11.0-561737.meta'

    i = 0
    for dir in glob.glob(input_dir):
        i += 1
        if i==4 or i==6: continue
        # if i not in [8, 9]: continue
        print('============================================================================================', dir)
        get_predict(dir, model_dir, meta_name)
        evaluation = Eval(model_dir, dir, threshhole=0.8, onset_tolerance=1)
        evaluation.frameF()
        evaluation.precision()
        index = 2000
        # evaluation.plot(index, index+1000)
        # evaluation.analysis()
        print('============================================================================================', dir)
    print()
                                                        