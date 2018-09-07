#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import os
import random
import mysql
import time
import pymysql
from sklearn import preprocessing

sr = 22050
step = 1        # times of window_size
window_size = 20       # ms

class DataGen:
    def __init__(self, batch_size=128, split=0.99):
        print('>>>>>>>>>>>>>>>>>> getting data......')
        self.__db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="polyphonic",port=3306)
        self.__cur = self.__db.cursor()
        self._batch_size = batch_size
        self.__index = mysql.get_index(self.__cur)

        random.shuffle(self.__index)

        self.__split = math.floor(split*len(self.__index))
        self.__train = self.__index[:self.__split]
        self.__val = self.__index[self.__split:]
        print('>>>>>>>>>>>>>>>>>> lens: ', len(self.__index),len(self.__train), len(self.__val))
        
        self.__gen_index_train = 0
        self.__gen_index_val = 0

    def get_train_len(self):
        return len(self.__train)
    def get_val_len(self):
        return len(self.__val)
    def train_steps(self):
        return math.ceil(len(self.__train)/self._batch_size)
    def val_steps(self):
        return math.ceil(len(self.__val)/self._batch_size)

    def train_gen(self):
        while True:
            if (self.__gen_index_train + 1) * self._batch_size > len(self.__train):
                frames = self.__train[self.__gen_index_train * self._batch_size:]
                self.__gen_index_train = 0
                random.shuffle(self.__train)
            else:
                frames = self.__train[self.__gen_index_train * self._batch_size:(self.__gen_index_train + 1) * self._batch_size]
                self.__gen_index_train += 1
            yield frames

    def val_gen(self):
        while True:
            if (self.__gen_index_val + 1) * self._batch_size > len(self.__val):
                frames = self.__val[self.__gen_index_val * self._batch_size:]
                self.__gen_index_val = 0
            else:
                frames = self.__val[self.__gen_index_val * self._batch_size:(self.__gen_index_val + 1) * self._batch_size]
                self.__gen_index_val += 1
            yield frames

    def get_param(self):
        return len(self.__train), len(self.__val), self.train_steps(), self.val_steps()

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
    data = DataGen()
    print(data.get_param())