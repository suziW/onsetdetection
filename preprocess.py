#!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import pretty_midi
import librosa
import librosa.display
import math

sr = 22050
step = 1/3        # times of window_size
window_size = 60       # ms
min_midi = 21
max_midi = 108
note_range = 88
midinote = 72       # middle C 60  e?76
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple

class Preprocess:
    def __init__(self, input_dir, output_dir):
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self.__step = math.floor(self.__window_size*step)
        self.__input_dir = input_dir
        self.__output_dir = output_dir

        self.__wavfiles = []
        self.__midfiles = []
        self.__input_num = 0
        self.__get_file()

        self.__y_groundtruth = []
        self.__y_input = []
        self.__align_list = []
        self.__x_input = []
        self.__midfile2np()
        self.__wavfile2np()
        # print(len(self.__x_input), len(self.__y_input))

        self.__save()
    
    def __save(self):
        print('----------- saving......')
        # indices = np.random.permutation(len(self.__x_input))
        self.__x_input = np.array(self.__x_input)
        self.__y_input = np.array(self.__y_input)
        self.__y_groundtruth = np.array(self.__y_groundtruth)
        assert(self.__x_input.shape[0]==self.__y_input.shape[0])
        # self.__x_input = self.__x_input[indices]
        # self.__y_input = self.__y_input[indices]
        mmx = np.memmap(filename=self.__output_dir+'x_input.dat', mode='w+', shape=self.__x_input.shape, dtype=float)
        mmx[:] = self.__x_input[:]
        mmy = np.memmap(filename=self.__output_dir+'y_input.dat', mode='w+', shape=self.__y_input.shape)
        mmy[:] = self.__y_input[:]
        mmgroundtruth = np.memmap(filename=self.__output_dir+'y_groundtruth.dat', mode='w+', shape=self.__y_groundtruth.shape)
        mmgroundtruth[:] = self.__y_groundtruth[:]
        print('>>>>>>>>>> (x, y, g).shape: ', mmx.shape, mmy.shape, mmgroundtruth.shape)
        del mmx, mmy, mmgroundtruth


    def __midfile2np(self):
        for file in self.__midfiles:
            midobj = pretty_midi.PrettyMIDI(file)     # loadfile
            # endtime = midobj.get_end_time()
            # print(endtime)
            mid_org = midobj.get_piano_roll(fs=sr)[min_midi:max_midi + 1].T #get_piano_roll ----> [notes, samples]
            print('>>>>>>>>>> mid_org:', file, mid_org.shape)
            mid = np.zeros(mid_org.shape)
            mid[mid_org > 0] = 1
            # for i, j in enumerate(np.sum(mid, axis=1)):
            #     if j>0.1:
            #         mid_org=mid_org[i:]
            #         mid = mid[i:]
            #         break
            print('>>>>>>>>>>> mid:', file, mid.shape)
            for i in np.arange(0, mid.shape[0]-self.__window_size+1, self.__step):
                onoff_detected = 0
                for note in range(note_range):
                    if mid[i+self.__step, note] < mid[i+self.__step*2, note]:
                        onoff_detected = 1
                self.__y_input.append(onoff_detected) 
                self.__y_groundtruth.append(mid_org[i+self.__step*2])
            self.__align_list.append(len(self.__y_input))
            # break
            

    def __wavfile2np(self):
        alignIndex = 0
        for file in self.__wavfiles:
            wav, _ = librosa.load(file, sr)
            print('>>>>>>>>>> wav: ', file, wav.shape)
            for i in np.arange(0, len(wav)-self.__window_size+1, self.__step):
                self.__x_input.append(wav[i:i+self.__window_size])
            self.__x_input = self.__x_input[:self.__align_list[alignIndex]]
            alignIndex += 1
            # break
            
        
    def __get_file(self):
        for wavfile in glob.glob(self.__input_dir+'*.wav'):
            self.__wavfiles.append(wavfile)
        for midfile in glob.glob(self.__input_dir+'*.mid'):
            self.__midfiles.append(midfile)
        self.__wavfiles.sort()
        self.__midfiles.sort()
        self.__input_num = len(self.__wavfiles)
        for i in range(len(self.__wavfiles)):
            assert(os.path.splitext(self.__wavfiles[i])[0] == os.path.splitext(self.__midfiles[i])[0]) 

    def get_param(self): 
        return {'input_num': self.__input_num, 'window_size': self.__window_size, 
                'step': self.__step, 'frame/ms': self.__framepms, 'x_input': self.__x_input.shape,
                'y_input': self.__y_input.shape} 

def plot(dir, begin, end):  # x_input': (27256, 1320), 'y_input': (27256,)
    mmy_groundtruth = np.memmap(dir + 'y_groundtruth.dat', mode='r')
    y_groundtruth = np.reshape(mmy_groundtruth, (-1, note_range))
    mm_y_input = np.memmap(dir + 'y_input.dat', mode='r')
    y_input = mm_y_input.reshape(1, -1)
    y_input = np.pad(y_input, ((87, 0), (0, 0)), 'maximum')
    mm_x_input = np.memmap(dir + 'x_input.dat', mode='r', dtype=float)
    x_input = np.reshape(mm_x_input, (-1, 1320))

    print('shapes: ', y_groundtruth.shape, y_input.shape, x_input.shape)
    print('onsets: ', sum(mm_y_input))
    for i in range(27256):
        if mm_y_input[i] == 1:
            print(i)
            plt.figure()
            plt.plot(x_input[i])
            plt.figure()
            plt.pcolor(y_groundtruth.T[:, begin:end]+y_input[:, begin:end]*10)
            plt.show()

if __name__=='__main__':
    input_dir = 'data/maps/train/'
    output_dir = 'data/maps/train/'

    pre = Preprocess(input_dir, output_dir)
    print(pre.get_param())
    plot(output_dir, 0, 500)