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

sr = 25600
step = 1        # times of window_size
window_size = 60       # ms
min_midi = 21
max_midi = 108
note_range = 88
midinote = 72       # middle C 60  e?76
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple
hop_length = 128

class Preprocess:
    def __init__(self, input_dir):
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = 12
        self.__step = math.floor(self.__window_size*step)
        self.__input_dir = input_dir
        self.__output_dir = input_dir
        self.__extra_onset_step = 5 * self.__framepms

        self.__wavfiles = []
        self.__midfiles = []
        self.__input_num = 0
        self.__get_file()

        self.__y_groundtruth = []
        self.__y_input = []
        self.__align_list = []
        self.__x_input = []
        self.__get_aligned_xy()
        # self.__midfile2np()
        # self.__wavfile2np()
        # print(len(self.__x_input), len(self.__y_input))

        self.__save()
    def __save(self):
        print('----------- saving......')
        self.__x_input = np.concatenate(self.__x_input)
        self.__y_input = np.concatenate(self.__y_input)
        # self.__y_groundtruth = np.array(self.__y_groundtruth)
        print('>>>>>>>>>>>>>>>>>>>>>>>> input.shape', self.__y_input.shape, self.__x_input.shape)
        assert(self.__x_input.shape[0]==self.__y_input.shape[0])
        mmx = np.memmap(filename=self.__output_dir+'x_input.dat', mode='w+', shape=self.__x_input.shape, dtype=float)
        mmx[:] = self.__x_input[:]
        mmy = np.memmap(filename=self.__output_dir+'y_input.dat', mode='w+', shape=self.__y_input.shape)
        mmy[:] = self.__y_input[:]
        # mmgroundtruth = np.memmap(filename=self.__output_dir+'y_groundtruth.dat', mode='w+', shape=self.__y_groundtruth.shape)
        # mmgroundtruth[:] = self.__y_groundtruth[:]
        # del mmgroundtruth, self.__y_groundtruth
        # print('>>>>>>>>>> (x, y, g).shape: ', mmx.shape, mmy.shape, mmgroundtruth.shape)
        # del mmx, mmy, mmgroundtruth

    def __get_aligned_xy(self):
        for file_cnt, mid_file in enumerate(self.__midfiles):
            print('>>>>>>>>>>>>>>>>>>>>>>>>{}/{} file {}'.format(file_cnt+1, len(self.__midfiles), mid_file))
            wav_file = os.path.splitext(mid_file)[0] + '.wav'
            wav_name = os.path.split(wav_file)[1]
            x_input = self.__wavfile2np(wav_file)
            print('>>>>>>>>>>>>>>>>>>>>>>>> x_input.shape', x_input.shape)

            times = librosa.frames_to_time(np.arange(x_input.shape[0])*self.__step, sr=sr, hop_length=hop_length)    # -wav_start_times[wav_name]
            y_input =  self.__midfile2np(mid_file, times)
            
            self.__y_input.append(y_input)
            self.__x_input.append(x_input)

            # times = [123, 543, 854, 1234, 4321, 6432, 9012]
            # for time in times:
            #     print(np.where(y_input[time]==1))
            #     for i in range(3):
            #         s = x_input[time, :, :, i]
            #         print(s.shape)
            #         s = s.T
            #         plt.figure()
            #         librosa.display.specshow(s, sr=sr, hop_length=hop_length, fmin=librosa.midi_to_hz(min_midi),
            #                             fmax=librosa.midi_to_hz(max_midi), bins_per_octave=bins_per_octave,y_axis='cqt_note', x_axis='time')
            #         plt.show()


    def __midfile2np(self, mid_file, times):
        midobj = pretty_midi.PrettyMIDI(mid_file)     # loadfile
        piano_roll = midobj.get_piano_roll(fs=sr, times=times)[min_midi:max_midi + 1].T
        piano_roll[piano_roll > 0] = 1
        return piano_roll

    def __wavfile2np(self, wav_file):
        y, _ = librosa.load(wav_file, sr)
        print('>>>>>>>>>>>>>>>>>>>>>>>> wav.shape', y.shape)
        S2 = librosa.hybrid_cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=2)
        print('----------1/3', end='\r')
        S3 = librosa.hybrid_cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=2)
        print('----------2/3', end='\r  ')
        S1 = librosa.hybrid_cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=0.5)
        print('----------3/3')

        # print('>>>>>>>>>>>>>>>>>>>>> S.shape: ', S2.shape)
        # plt.figure()
        # librosa.display.specshow(S2[:, 2000:4000], sr=sr, hop_length=hop_length, fmin=librosa.midi_to_hz(min_midi),
        #                     fmax=librosa.midi_to_hz(max_midi), bins_per_octave=bins_per_octave,y_axis='cqt_note', x_axis='time')
        # plt.show()

        S1 = self.__s_process(S1)
        S2 = self.__s_process(S2)
        S3 = self.__s_process(S3)

        S = np.array([S1, S2, S3])
        # print('>>>>>>>>>>>>>>>>>>>>> S.shape: ', S.shape)
        S = S.transpose(1, 2, 0)
        print('>>>>>>>>>>>>>>>>>>>>>>>> S.shape', S.shape)
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
        for wavfile in glob.glob(self.__input_dir+'*.wav'):
            self.__wavfiles.append(wavfile)
        for midfile in glob.glob(self.__input_dir+'*.mid'):
            self.__midfiles.append(midfile)
        print('>>>>>>>>>>>>>>>>>>>>>>>> total files', len(self.__midfiles))
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
            plt.pcolor(y_groundtruth.T[:, begin:end]+y_input[:, begin:end]*20)
            plt.show()

if __name__=='__main__':
    input_dir = '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_StbgTGd2_2/*/MUS/'
    i = 0
    for dir in glob.glob(input_dir):
        i += 1
        # if i != 3: continue
        print(dir)
        pre = Preprocess(dir)
        # print(pre.get_param())
        # plot(dir, 0, 2000)