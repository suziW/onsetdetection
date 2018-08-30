#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import os
import random

sr = 22050
step = 1/3        # times of window_size
window_size = 60       # ms

class DataGen:
    def __init__(self, dirpath, batch_size=128, split=1000):
        print('>>>>>>>>>>>>>>>>>> getting data from dirpath: ', dirpath)       
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self._batch_size = batch_size

        self._dirx = os.path.join(dirpath, 'x_input.dat')
        self._diry = os.path.join(dirpath, 'y_input.dat')

        self.__x_inputs = np.array(0)
        self.__y_inputs = np.array(0)
        self.__readmm()
        
        # self.shuffle_inputs()
        self.__split = -split
        self.__x_train = self.__x_inputs[:self.__split]
        self.__y_train = self.__y_inputs[:self.__split]
        self.__x_test = self.__x_inputs[self.__split:]
        self.__y_test = self.__y_inputs[self.__split:]
        # self.combat_imbalance()
        self.i = 0
    def combat_imbalance(self):
        print('>>>>>>>>>>>>>>>>>> in combat......')
        train_len = self.__y_train.shape[0]
        train_ones = np.sum(self.__y_train)
        index = list(filter(lambda i:self.__y_train[i]==1, (i for i in range(train_len))))
        index_append = []
        cnt = math.floor(train_len/train_ones) - 2
        for _ in range(cnt):
            index_append+=index
        index_append+=range(train_len)
        self.__y_train = self.__y_train[index_append]
        self.__x_train = self.__x_train[index_append]
        self.shuffle_train()
        print('>>>>>>>>>>>>>>>>>> combat info: trainlen trianones indexlen cnt', train_len, train_ones, len(index_append), cnt)


    def shuffle_inputs(self):
        assert(self.__x_inputs.shape[0]==self.__y_inputs.shape[0])
        indices = np.random.permutation(self.__y_inputs.shape[0])
        self.__x_inputs = self.__x_inputs[indices]
        self.__y_inputs = self.__y_inputs[indices]
    def shuffle_train(self):
        assert(self.__x_train.shape[0]==self.__y_train.shape[0])
        indices = np.random.permutation(self.__y_train.shape[0])
        self.__x_train = self.__x_train[indices]
        self.__y_train = self.__y_train[indices]
    def get_test_data(self):
        return self.__x_test, self.__y_test.reshape(-1, 1)
    def getinfo_train(self):
        return self.__x_train.shape, self.__y_train.shape, np.sum(self.__y_train)
    def getinfo_test(self):
        return self.__x_test.shape, self.__y_test.shape, np.sum(self.__y_test)
    def train_steps(self):
        return math.ceil(self.__x_train.shape[0]/self._batch_size)
    def test_steps(self):
        return math.ceil(self.__x_test.shape[0]/self._batch_size)
    def train_gen(self):
        while True:
            if (self.i + 1) * self._batch_size > self.__x_train.shape[0]:
                # return rest and then switch files
                x, y = self.__x_train[self.i * self._batch_size:], self.__y_train[self.i * self._batch_size:]
                self.i = 0
                self.shuffle_train()
            else:
                x, y = self.__x_train[self.i * self._batch_size:(self.i + 1) * self._batch_size],\
                        self.__y_train[self.i * self._batch_size:(self.i + 1) * self._batch_size]
                self.i += 1
            # y = y.reshape(-1, 1)
            yield x, y

    def test_gen(self):
        while True:
            if (self.i + 1) * self._batch_size > self.__x_test.shape[0]:
                # return rest and then switch files
                x, y = self.__x_test[self.i * self._batch_size:], self.__y_test[self.i * self._batch_size:]
                self.i = 0
            else:
                x, y = self.__x_test[self.i * self._batch_size:(self.i + 1) * self._batch_size],\
                        self.__y_test[self.i * self._batch_size:(self.i + 1) * self._batch_size]
                self.i += 1
            # y = y.reshape(-1, 1)
            yield x, y

    def __readmm(self):
        mmx = np.memmap(self._dirx, mode='r', dtype=float)
        self.__x_inputs = mmx.reshape(-1, self.__window_size)
        mmy = np.memmap(self._diry, mode='r')
        self.__y_inputs = mmy
        # print('inputs shape:', self.__x_inputs.shape, self.__y_inputs.shape) 
        assert(self.__x_inputs.shape[0]==self.__y_inputs.shape[0])
        del mmx
        del mmy
    def check_data(self, x, y, nfft=4096, midinote = 72):
        print('@@@@@@@@@-----{}:{}-----{}/fftdot--@@@@@@@@@@'.format(librosa.midi_to_hz(midinote), 
                                    librosa.midi_to_hz(midinote)/sr*nfft, sr/nfft))
        if y==1:
            plt.figure()
            plt.plot(x)
            plt.figure()
            transformed = np.fft.fft(x, n=nfft)
            plt.plot(abs(transformed))
            plt.show()

if __name__ == '__main__':
    dirpath = 'data/'

    dataGen = DataGen(dirpath, batch_size=1, split=1)
    # dataGen.combat_imbalance()
    sum = 0
    temp = 0
    for i in range(20000):
        x, y = next(dataGen.train_gen())
        if y>0:
            dataGen.check_data(x[0], y[0])
    print(sum)