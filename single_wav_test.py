#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import librosa
import librosa.display
import tensorflow as tf 
from sklearn import preprocessing
import math

sr = 22050
window_size = 1320       
step = math.floor(1320/3)        # times of window_size
min_midi = 21
max_midi = 108
note_range = 88
midinote = 72       # middle C 60  e?76
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple
data_dir = 'data/piano/'

def data2dat():
    x_input = []
    for wavfile in glob.glob(data_dir+'*.wav'):
        print('>>>>>>>>>> wavfile:', wavfile)
        wav, _ = librosa.load('data/piano/alb_esp4_format0.wav', sr)
        print('>>>>>>>>>> wav: ', wavfile, wav.shape)
        for i in np.arange(0, len(wav)-window_size+1, step):
            x_input.append(wav[i:i+window_size])
    x_input = np.array(x_input)
    print('>>>>>>>>>> xinput shape: ', x_input.shape)
    mmx = np.memmap(filename=data_dir+'x_input.dat', mode='w+', shape=x_input.shape, dtype=float)
    mmx[:] = x_input[:]
    del mmx

class Datagen:
    def __init__(self, batch_size):
        mmx = np.memmap(data_dir+'x_input.dat', mode='r', dtype=float)
        self.x_input = mmx.reshape(-1, window_size)
        self.batch_size = batch_size
        self.i = 0
    def datagen(self):
        while True:
            if (self.i + 1) * self.batch_size > self.x_input.shape[0]:
                # return rest and then switch files
                x = self.x_input[self.i * self.batch_size:]
                self.i = 0
            else:
                x = self.x_input[self.i * self.batch_size:(self.i + 1) * self.batch_size]
                self.i += 1
            yield x
    def steps(self):
        return math.ceil(self.x_input.shape[0]/self.batch_size)


def get_predict(model_dir, meta_name):
    data = Datagen(32)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_dir, 'savers/', meta_name))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(model_dir, 'savers/')))

        graph = tf.get_default_graph()
        X = graph.get_operation_by_name('x_input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        prediction = tf.get_collection('pred_collection')[0]

        y_prediction = []
        for i in range(data.steps()):
            batch_x = next(data.datagen())
            # batch_x = batch_x * 10
            batch_x = preprocessing.StandardScaler().fit_transform(batch_x.T).T 
            print('predicting.........................................{}/{}.......'.format(i, data.steps()), end='\r')
            pred_batch = sess.run(prediction, feed_dict={X: batch_x, keep_prob: 1})
            y_prediction.append(pred_batch)
            # break
        y_prediction = np.concatenate(y_prediction)

        print('>>>>>>>>>>>>>>>>>> saving: ', data_dir)
        mmy = np.memmap(filename=os.path.join(data_dir, 'y_pred.dat'), mode='w+', shape=y_prediction.shape[0], dtype=float)
        mmy[:] = y_prediction[: ,0]
        del mmy

class Eval:
    def __init__(self, threshhole=0.5):
        mmy_pred = np.memmap(os.path.join(data_dir, 'y_pred.dat'), mode='c', dtype=float)
        self.y_pred = mmy_pred.reshape(1, -1)
        self.y_pred[self.y_pred>=threshhole] = 1
        self.y_pred[self.y_pred<threshhole] = 0

    def plot(self, begin, end): 
        plt.figure()
        plt.pcolor(self.y_pred[:, begin:end])
        plt.title('pred onset')
        x_lable = list(range(begin, end, 15))
        x_reject = [round((x*0.02+0.03), 2) for x in x_lable]
        plt.xticks(x_lable, x_reject)
        plt.show()

if __name__=='__main__':
    # data2dat()

    # model_dir = 'model/'
    # meta_name = '2-4.0-6164.meta'
    # get_predict(model_dir, meta_name)

    eval = Eval(threshhole=0.8)
    eval.plot(0, 500)