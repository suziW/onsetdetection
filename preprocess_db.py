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
import random

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
    def __init__(self, input_dir):
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self.__step = int(window_size*step*self.__framepms)
        self.__input_dir = input_dir
        self.__wav_mid_delta = 1 * self.__framepms
        self.__extra_onset = 4

        self.__wavfiles = []
        self.__midfiles = []
        self.__input_num = 0
        self.__get_file()

        self.__y_input = []
        self.__align_list = []
        self.__x_input = []
        self.__get_aligned_xy()

        # self.__save()
    
    def __save(self):
        print('----------- saving......')
        assert(len(self.__x_input) == len(self.__y_input))
        length = len(self.__y_input)
        db = pymysql.connect(host="localhost",user="root",
            password="1234",db="onset_detection",port=3306)
        cur = db.cursor()
        sql = "insert into maps_final(x_train, y_onset) values(%s, %s)"
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
            print('>>>>>>>>>>>>>>>>>>>>>>>> file', mid_file)
            wav_file = os.path.splitext(mid_file)[0] + '.wav'
            wav_name = os.path.split(wav_file)[1]
            self.__x_input = self.__wavfile2np(wav_file)

            times = librosa.frames_to_time(np.arange(self.__x_input.shape[0]), sr=sr, hop_length=hop_length)    # -wav_start_times[wav_name]
            print('>>>>>>>>>>>>>>>>>>>>>>>> x_input.shape', self.__x_input.shape)
            self.__y_input = self.__midfile2np(mid_file, times)
            print('>>>>>>>>>>>>>>>>>>>>>>>> y_input.shape', self.__y_input.shape)



    def __midfile2np(self, mid_file, times):
        midobj = pretty_midi.PrettyMIDI(mid_file)     # loadfile
        piano_roll = midobj.get_piano_roll(fs=sr, times=times)[min_midi:max_midi + 1].T
        piano_roll[piano_roll > 0] = 1
        return piano_roll

    def __wavfile2np(self, wav_file):
        y, _ = librosa.load(wav_file, sr)
        print('>>>>>>>>>>>>>>>>>>>>>>>> y.shape: ', y.shape)
        S2 = librosa.hybrid_cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=1)
        print('----------', end='>')
        S3 = librosa.hybrid_cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=2)
        print('----------', end='>')
        S1 = librosa.hybrid_cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=0.5)
        print('----------')
        # print('>>>>>>>>>>>>>>>>>>>>> S.shape: ', S2.shape)

        S1 = self.__s_process(S1)
        S2 = self.__s_process(S2)
        S3 = self.__s_process(S3)

        S = np.array([S1, S2, S3])
        # print('>>>>>>>>>>>>>>>>>>>>> S.shape: ', S.shape)
        S = S.transpose(1, 2, 0)
        print('>>>>>>>>>>>>>>>>>>>>>>>> S.shape: ', S.shape)
        # print(np.min(S2), np.max(S2), np.mean(S2))

        windows = []
        for i in range(0, S.shape[0] - self.__window_size + 1, self.__step):
            w = S[i:i + self.__window_size, :]
            windows.append(w)

        # print inputs
        x = np.array(windows)
        # print(x.shape)
        # print(x[0].shape)
        return x

    def __s_process(self, s):
        s = np.abs(s.T)
        minDB = np.min(s)
        s = np.pad(s, ((self.__window_size//2, self.__window_size//2), (0, 0)), 'constant', constant_values=minDB)
        return s
            
        
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