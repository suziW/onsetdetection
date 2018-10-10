#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

from input_queue import InputGen
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

min_midi = 21
max_midi = 108
note_range = 88
sr = 22050
pic_dir = '/media/admin1/32B44FF2B44FB75F/pic'

queue =  InputGen(batch_size=100, split=0.9, thread_num=5)
fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
ax = fig.add_subplot(111)

for i in range(5000):
    print('====================================== {}/5000'.format(i), end='\r')
    train = next(queue.train_gen())
    lables = train[1]
    pics = train[0]

    count = 0
    for pic in pics:
        S = librosa.hybrid_cqt(pic, fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=128,
                                bins_per_octave=4*12,  n_bins=88*4, filter_scale=1)
        # plt.figure()
        # librosa.display.specshow(S, sr=sr, fmin=librosa.midi_to_hz(min_midi),
        #                             fmax=librosa.midi_to_hz(max_midi), y_axis='linear')
        # plt.show()

        plt.pcolormesh(np.abs(S))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0, 0)

        plt.savefig('{}/train-{}-{}.jpg'.format(pic_dir, 100*i + count, lables[count]))
        count += 1

        # plt.show()


for i in range(500):
    print('====================================== {}/500'.format(i), end='\r')
    val = next(queue.val_gen())
    lables = val[1]
    pics = val[0]

    count = 0
    # fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    # ax = fig.add_subplot(111)
    for pic in pics:
        S = librosa.hybrid_cqt(pic, fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=128,
                                bins_per_octave=4*12,  n_bins=88*4, filter_scale=1)
        # plt.figure()
        # librosa.display.specshow(S, sr=sr, fmin=librosa.midi_to_hz(min_midi),
        #                             fmax=librosa.midi_to_hz(max_midi), y_axis='linear')
        # plt.show()

        plt.pcolormesh(np.abs(S))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0, 0)

        plt.savefig('{}/val-{}-{}.jpg'.format(pic_dir, 100*i + count, lables[count]))
        count += 1
        # plt.show()