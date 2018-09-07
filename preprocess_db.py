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
import pymysql
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
        self.__step = int(window_size*step*self.__framepms)
        self.__input_dir = input_dir

        self.__wavfiles = []
        self.__midfiles = []
        self.__input_num = 0
        self.__get_file()

        self.__y_input = []
        self.__x_input = []
        self.__get_aligned_xy()

        # self.__save()
    
    def __save(self):
        print('----------- saving......')
        assert(len(self.__x_input) == len(self.__y_input))
        length = len(self.__y_input)
        db = pymysql.connect(host="localhost",user="root",
            password="1234",db="polyphonic",port=3306)
        cur = db.cursor()
        sql = "insert into maps(x_input, y_input) values(%s, %s)"
        try:
            cur.executemany(sql, [(self.__x_input[i], self.__y_input[i]) for i in range(length)])
            # cur.executemany(sql, self.__x_input)
            # cur.execute(sql, self.__x_input[111])
        except Exception as e:
            db.rollback()
            print('except', e)
        db.commit()
        db.close()
        self.__x_input = []
        self.__y_input = []

    def __get_aligned_xy(self):
        for mid_file in self.__midfiles:
            wav_file = os.path.splitext(mid_file)[0] + '.wav'
            wav_name = os.path.split(wav_file)[1]
            # get mid
            midobj = pretty_midi.PrettyMIDI(mid_file)     # loadfile
            mid = midobj.get_piano_roll(fs=sr)[min_midi:max_midi + 1].T #get_piano_roll ----> [notes, samples]
            mid[mid>0] = 1
            mid = mid.astype(np.int8)
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
                self.__y_input.append(mid[int(i+self.__window_size/2)].tobytes()) 
                self.__x_input.append(wav[i:i+self.__window_size].tobytes())
            self.__save()
                # break
        
    def __get_file(self):
        print(self.__input_dir+'*.wav')
        for wavfile in glob.glob(self.__input_dir+'*.wav'):
            self.__wavfiles.append(wavfile)
        for midfile in glob.glob(self.__input_dir+'*.mid'):
            self.__midfiles.append(midfile)
        self.__wavfiles.sort()
        self.__midfiles.sort()
        self.__input_num = len(self.__wavfiles)
        print('>>>>>>>>>>> total files: ', self.__input_num)
        # print(self.__wavfiles)
        for i in range(len(self.__wavfiles)):
            assert(os.path.splitext(self.__wavfiles[i])[0] == os.path.splitext(self.__midfiles[i])[0]) 

    def get_param(self): 
        return {'input_num': self.__input_num, 'window_size': self.__window_size, 
                'step': self.__step, 'frame/ms': self.__framepms} 


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
    input_dir = [
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnBcht_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnBsdf_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnCGdD_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnStgb_2/*/MUS/',
        # '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_ENSTDkAm_2/*/MUS/',
        # '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_ENSTDkCl_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_SptkBGAm_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_SptkBGCl_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_StbgTGd2_2/*/MUS/'
    ]
    for i in range(9):
        print('==================================================================================== i = ', i)
        pre = Preprocess(input_dir[i])
        print(pre.get_param())
        del pre
        print('==================================================================================== i = ', i)
    # plot(output_dir, 0, 500)
    # input_dir = 'data/alb/'
    # pre = Preprocess(input_dir)
    # print(pre.get_param())