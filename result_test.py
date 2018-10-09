#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import glob
import tensorflow as tf
import load
import numpy as np 
import os 
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

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

        mmy_onset = np.memmap(os.path.join(model_dir, 'y_onset.dat'), mode='r')
        self.y_onset = mmy_onset
        mmy_pred = np.memmap(os.path.join(model_dir, 'y_pred.dat'), mode='c', dtype=float)
        self.y_pred = mmy_pred
        self.__prob2onehot()

        mmy_groundtruth = np.memmap(input_dir + 'y_groundtruth.dat', mode='r')
        self.y_groundtruth = mmy_groundtruth.reshape(-1, 88).T
        self.y_onset_pad = np.pad(self.y_onset.reshape(1, -1), ((87, 0), (0, 0)), 'maximum')
        self.y_pred_pad = np.pad(self.y_pred.reshape(1, -1), ((87, 0), (0, 0)), 'maximum')

        del mmy_onset
        del mmy_pred
        del mmy_groundtruth
        print('>>>>>>>>>>>>>>>>>> shapes: ', self.y_onset.shape, self.y_pred.shape, self.y_groundtruth.shape,
                                         self.y_onset_pad.shape, self.y_pred_pad.shape)
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
        false_index = []
        true_index = []
        for i in range(len(self.y_onset)):
            if self.y_pred[i] == 1:
                total += 1
                if np.sum(self.y_onset[i-self.__onset_tolerance:i+1+self.__onset_tolerance]) > 0:
                    true_index.append(i) 
                else:
                    false_index.append(i)
        print('-----------------precision: {}/{}={}'.format(len(true_index), total, len(true_index)/total))
        print('-----------------total false is {}'.format(len(false_index)))
        # print('-----------------scores predict to one, actually zero: ')
        # false_score = [self.y_pred[i] for i in false_index]
        # print(false_score)
        return len(true_index)/total

    def frameR(self):       # how many groundtruth are predicted right
        total = 0
        false_index = []
        true_index = []
        for i in range(len(self.y_onset)):
            if self.y_onset[i] == 1:
                total += 1
                if np.sum(self.y_pred[i-self.__onset_tolerance:i+1+self.__onset_tolerance]) > 0:
                    true_index.append(i) 
                else:
                    false_index.append(i)
        print('-----------------recall: {}/{}={}'.format(len(true_index), total, len(true_index)/total))
        print('-----------------total false is {}'.format(len(false_index)))
        # print('-----------------scores should be one, but predict to zero: ')
        # false_score = [self.y_pred[i] for i in false_index]
        # print(false_score)    
        return len(true_index)/total

    def frameF(self):
        precision = self.frameP()
        recall = self.frameR()
        print('-----------------f measure is: ', precision*recall*2/(precision+recall))
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
    model_dir = 'model/'
    meta_name = '0.980809691686963-48.0-2451216.meta'

    i = 0
    for dir in glob.glob(input_dir):
        i += 1
        # if i !=3: continue
        print('============================================================================================', dir)
        get_predict(dir, model_dir, meta_name)
        evaluation = Eval(model_dir, dir, threshhole=0.5, onset_tolerance=0)
        evaluation.frameF()
        evaluation.precision()
        index = 0
        # evaluation.plot(index, index+1000)
        print('============================================================================================', dir)
