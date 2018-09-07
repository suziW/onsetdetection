#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import os
import random
import glob 

sr = 22050
step = 1        # times of window_size
window_size = 20       # ms

class DataGen:
    def __init__(self, dirpath, batch_size=128):
        print('>>>>>>>>>>>>>>>>>> getting data from dirpath: ', dirpath)
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self._batch_size = batch_size

        self._dirx = os.path.join(dirpath, 'polyphonic/x_input.dat')
        self._diry = os.path.join(dirpath, 'polyphonic/y_input.dat')

        self.__x_inputs = np.array(0)
        self.__y_inputs = np.array(0)
        self.__readmm()
        
        self.i = 0
    def get_info(self):
        return self.get_test_len(), self.test_steps()
    def get_test_len(self):
        return self.__x_inputs.shape[0]
    def test_steps(self):
        return math.ceil(self.__x_inputs.shape[0]/self._batch_size)
    def test_gen(self):
        while True:
            if (self.i + 1) * self._batch_size > self.__x_inputs.shape[0]:
                # return rest and then switch files
                x, y = self.__x_inputs[self.i * self._batch_size:],\
                        self.__y_inputs[self.i * self._batch_size:]
                self.i = 0
            else:
                x, y = self.__x_inputs[self.i * self._batch_size:(self.i + 1) * self._batch_size],\
                        self.__y_inputs[self.i * self._batch_size:(self.i + 1) * self._batch_size]
                self.i += 1
            # y = y.reshape(-1, 1)
            yield x, y

    def __readmm(self):
        mmx = np.memmap(self._dirx, mode='r', dtype=float)
        self.__x_inputs = mmx.reshape(-1, self.__window_size)
        mmy = np.memmap(self._diry, mode='r')
        self.__y_inputs = mmy.reshape(-1, 88)
        # print('inputs shape:', self.__x_inputs.shape, self.__y_inputs.shape) 
        assert(self.__x_inputs.shape[0]==self.__y_inputs.shape[0])
        del mmx
        del mmy

    def check_data(self, x, y, nfft=4096, midinote = 72):
        print('@@@@@@@@@-----{}:{}-----{}/fftdot--@@@@@@@@@@'.format(librosa.midi_to_hz(midinote),
                                    librosa.midi_to_hz(midinote)/sr*nfft, sr/nfft))
        plt.figure()
        plt.plot(x)
        plt.figure()
        transformed = np.fft.fft(x, n=nfft)
        plt.plot(abs(transformed))
        plt.show()

if __name__ == '__main__':
    dirpath = 'data/maps/test/*/'
    for dir in glob.glob(dirpath):
        dataGen = DataGen(dir, batch_size=1)
        # dataGen.combat_imbalance()
        sum = 0
        temp = 0
        for i in range(20000):
            x, y = next(dataGen.test_gen())
            if np.sum(y) == 1:
                dataGen.check_data(x[0], y[0], midinote=np.argmax(y)+21)
        print(sum)