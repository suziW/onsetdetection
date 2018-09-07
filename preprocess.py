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
from wav_start_time import wav_start_times

sr = 22050
step = 1        # times of window_size
window_size = 20       # ms
min_midi = 21
max_midi = 108
note_range = 88
midinote = 72       # middle C 60  e?76
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple

class Preprocess:
    def __init__(self, input_dir):
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self.__step = math.floor(self.__window_size*step)
        self.__input_dir = input_dir
        self.__output_dir = input_dir

        self.__wavfiles = []
        self.__midfiles = []
        self.__input_num = 0
        self.__get_file()

        self.__y_input = []
        self.__x_input = []
        self.__get_aligned_xy()

        self.__save()
    
    def __save(self):
        print('----------- saving......')
        self.__x_input = np.array(self.__x_input)
        self.__y_input = np.array(self.__y_input)
        assert(self.__x_input.shape[0]==self.__y_input.shape[0])
        mmx = np.memmap(filename=self.__output_dir+'polyphonic/x_input.dat', mode='w+', shape=self.__x_input.shape, dtype=float)
        mmx[:] = self.__x_input[:]
        # del mmx, self.__x_input
        mmy = np.memmap(filename=self.__output_dir+'polyphonic/y_input.dat', mode='w+', shape=self.__y_input.shape)
        mmy[:] = self.__y_input[:]
        # del mmy, self.__y_input

    def __get_aligned_xy(self):
        for mid_file in self.__midfiles:
            wav_file = os.path.splitext(mid_file)[0] + '.wav'
            wav_name = os.path.split(wav_file)[1]
            # get mid
            midobj = pretty_midi.PrettyMIDI(mid_file)     # loadfile
            mid = midobj.get_piano_roll(fs=sr)[min_midi:max_midi + 1].T #get_piano_roll ----> [notes, samples]
            mid[mid>0] = 1
            mid = mid.astype(np.int8)   # shape [samples, notes]
            for i, j in enumerate(np.sum(mid, axis=1)):
                if j>0.1:
                    mid = mid[i:]
                    break
            print('>>>>>>>>>>> mid:', mid_file, mid.shape)

            # get wav
            wav, _ = librosa.load(wav_file, sr)
            wav_start_time = int(wav_start_times[wav_name]*sr)
            wav = wav[wav_start_time:]
            print('>>>>>>>>>>> wav:', wav_file, wav.shape)

            for i in np.arange(0, mid.shape[0]-self.__window_size+1, self.__step):
                self.__y_input.append(mid[int(i+self.__window_size/2)]) # shape [frames, notes]
                self.__x_input.append(wav[i:i+self.__window_size])  # shape [frames, window_size]
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

def plot(dir, begin, end):
    mm_y_input = np.memmap(dir + 'polyphonic/y_input.dat', mode='r')
    y_input = mm_y_input.reshape(-1, note_range)
    mm_x_input = np.memmap(dir + 'polyphonic/x_input.dat', mode='r', dtype=float)
    x_input = np.reshape(mm_x_input, (-1, 440))
    
    plt.figure()
    plt.pcolor(y_input[begin:end, :].T)
    plt.show()

if __name__=='__main__':
    input_dir = 'data/maps/test/*/'
    i = 0
    for dir in glob.glob(input_dir):
        i += 1
        # if i != 3: continue
        print(dir)
        pre = Preprocess(dir)
        print(pre.get_param())
        plot(dir, 0, 1000)