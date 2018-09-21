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
    def __init__(self, batch_size=32, timesteps=20, split=0.99):
        print('>>>>>>>>>>>>>>>>>> getting data......')
        self.__db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="polyphonic",port=3306)
        self.__cur = self.__db.cursor()
        self.__index = mysql.get_index(self.__cur)

        self._batch_size = batch_size
        self._timesteps = timesteps

        self.__split = math.floor(split*len(self.__index))
        self.__train = self.__index[:self.__split]
        self.__val = self.__index[self.__split:]
        print('>>>>>>>>>>>>>>>>>> lens: ', len(self.__index),len(self.__train), len(self.__val))

        self._segment_train = len(self.__train) // self._batch_size
        self._segment_val = len(self.__val) // self._batch_size
        
        self._cursor_train = [ offset * self._segment_train for offset in range(self._batch_size)]
        self._cursor_val = [ offset * self._segment_val for offset in range(self._batch_size)]

    def get_train_len(self):
        return len(self.__train)
    def get_val_len(self):
        return len(self.__val)
    def train_steps(self):
        return math.ceil(len(self.__train)/(self._batch_size*self._timesteps))
    def val_steps(self):
        return math.ceil(len(self.__val)/(self._batch_size*self._timesteps))

    def train_gen(self):
        while True:
            frames = []
            for b in range(self._batch_size):
                for _ in range(self._timesteps):
                    frames.append(self.__train[self._cursor_train[b]])
                    self._cursor_train[b] = (self._cursor_train[b] + 1) % len(self.__train)
            yield frames        # size [batch_size * timesteps]

    def val_gen(self):
        while True:
            frames = []
            for b in range(self._batch_size):
                for _ in range(self._timesteps):
                    frames.append(self.__val[self._cursor_val[b]])
                    self._cursor_val[b] = (self._cursor_val[b] + 1) % len(self.__val)
            yield frames        # size [batch_size * timesteps]

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