
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
import random

sr = 25600
step = 1        # times of window_size
window_size = 1       # s
min_midi = 21
max_midi = 108
note_range = 88
hop_length = 512
midinote = 72       # middle C 60  e?76
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple
fig_size = 4.16 #

class Preprocess:
    def __init__(self, input_dir):
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = int(window_size*sr)
        self.__step = int(self.__window_size*step)
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
        self.__fig = plt.figure(figsize=(fig_size, fig_size), dpi=100)
        self.__bbox = []    # x, y, w, h
        self.__get_aligned_xy()



    def __get_aligned_xy(self):
        for mid_file in self.__midfiles:
            wav_file = os.path.splitext(mid_file)[0] + '.wav'
            wav_name = os.path.splitext(os.path.split(wav_file)[1])[0]
            # if not wav_name=='MAPS_MUS-bach_847_AkPnBcht': continue
            if not os.path.exists('pic/object_detection/{}'.format(wav_name)):
                print('making path pic/object_detection/{}'.format(wav_name))
                os.makedirs('pic/object_detection/{}'.format(wav_name))

            # get mid
            midobj = pretty_midi.PrettyMIDI(mid_file)     # loadfile
            mid = midobj.get_piano_roll(fs=sr)[min_midi:max_midi + 1].T #get_piano_roll ----> [notes, samples]
            onsets = (midobj.get_onsets()*sr).astype(int)
            onsets = np.unique(onsets).tolist()
            mid_len = mid.shape[0]
            print('>>>>>>>>>>> mid:', mid_file, mid.shape)

            # get wav
            wav, _ = librosa.load(wav_file, sr)
            print('>>>>>>>>>>> wav:', wav_file, wav.shape)

            for i in np.arange(0, mid_len-self.__window_size+1, self.__step):
####################################################################
                onset_detected = 0
                onset_location = []
                print('+++++++++++++====================>>>>>>>>>>>>>>i == onset[:10]',i , onsets[:10])
                for onset in onsets[:30]:
                    if onset < (i + self.__window_size):    # window size >= step
                        if onset >= i:
                            onset_detected = 1
                            onset_location.append(round((onset - i)/self.__window_size, 4))
                        if onset < (i + self.__step):
                            onsets.remove(onset)
####################################################################
                print('+++++++++++++====================>>>>>>>>>>>>>>onset_location', str(onset_location))
                if i-self.__window_size<0: continue
                if i+2*self.__window_size>len(wav): 
                    print('======================', '\n'*6, '======================')
                    continue
                x_input = wav[i-self.__window_size:i+2*self.__window_size]
                print('==============', x_input.shape)

                self.__fig.add_subplot(111)
                S = librosa.cqt(x_input, fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=128,
                                        bins_per_octave=4*12,  n_bins=88*4, filter_scale=0.3)[:, 200:401]
                plt.pcolormesh(np.abs(S), cmap='jet')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0, 0)
                plt.savefig('pic/object_detection/{}/{}--{}-cqt.jpg'.format(wav_name, i/sr, str(onset_location)))
                # plt.show()
                plt.clf()


                self.__fig.add_subplot(111)
                plt.pcolormesh(mid[np.arange(i, i+self.__window_size, 220)].T, vmin=0, vmax=50, cmap='jet')
                print('*'*55)
                print(S.shape, mid[np.arange(i, i+self.__window_size, 220)].T.shape)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0, 0)
                plt.savefig('pic/object_detection/{}/{}--{}-mid.jpg'.format(wav_name, i/sr, str(onset_location)))         
                # plt.show()
                plt.clf()


    def __wavfile2np(self):
        alignIndex = 0
        for file in self.__wavfiles:
            wav, _ = librosa.load(file, sr)
            print('>>>>>>>>>> wav: ', file, wav.shape)
            for i in np.arange(0, len(wav)-self.__window_size+1, self.__step):
                x_input = wav[i:i+self.__window_size].tobytes()
                self.__x_input.append(x_input)
                # break
            self.__x_input = self.__x_input[:self.__align_list[alignIndex]]
            alignIndex += 1
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

    def midi_list_to_string(self, note_list):   
        s = ''
        if sorted(list(note_list)) is None :      
            pass
        else:
            for note in sorted(list(note_list)) :
                s += '_M' + str(note)   
        return s
    

    def wav2inputnp(self, audio_fn,spec_type='cqt',bin_multiple=3):
        # print("wav2inputnp>>>>>>")
        bins_per_octave = 12 * bin_multiple #should be a multiple of 12
        n_bins = (max_midi - min_midi + 1) * bin_multiple
        S = librosa.cqt(audio_fn,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins)
        # print(S.shape)
        S = S.T
        S = np.abs(S)
        # print(np.min(S),np.max(S),np.mean(S))
        return S

if __name__=='__main__':
  
    input_dir = [      
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnBcht_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnBsdf_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnCGdD_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_AkPnStgb_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_ENSTDkAm_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_ENSTDkCl_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_SptkBGAm_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_SptkBGCl_2/*/MUS/',
        '/media/admin1/32B44FF2B44FB75F/Data/MAPS/MAPS_StbgTGd2_2/*/MUS/'
    ]
   
    for i in range(9):
        print('==================================================================================== i = ', i)
        pre = Preprocess(input_dir[i])
        print(pre.get_param())
        del pre
        break
        print('==================================================================================== i = ', i)
    # plot(output_dir, 0, 500)
    # input_dir = 'data/alb/'
    # pre = Preprocess(input_dir)
    # print(pre.get_param())

    def get_onsets(self):
        """Return a sorted list of the times of all onsets of all notes from
        all instruments.  May have duplicate entries.

        Returns
        -------
        onsets : np.ndarray
            Onset locations, in seconds.

        """
        onsets = np.array([])
        # Just concatenate onsets from all the instruments
        for instrument in self.instruments:
            onsets = np.append(onsets, instrument.get_onsets())
        # Return them sorted (because why not?)
        return np.sort(onsets)
