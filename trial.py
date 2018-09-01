import time
import pymysql
import mysql
import numpy as np
from sklearn import preprocessing
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt 
import os 
import glob

def plot(dir):
    mmy_onset = np.memmap(os.path.join(dir, 'y_input.dat'), mode='r')
    y_onset = mmy_onset
    del mmy_onset

    y_onset_pad = np.pad(y_onset.reshape(1, -1), ((87, 0), (2, 0)), 'edge')
    print('............. y_onset_pad:', y_onset_pad.shape)

    mmy_groundtruth = np.memmap(dir + 'y_groundtruth.dat', mode='r')
    y_groundtruth = mmy_groundtruth.reshape(-1, 88).T
    del mmy_groundtruth
    print('............. groundtruth:', y_onset_pad.shape)
    
    plt.figure(figsize=(88, 500))
    plt.pcolor(y_groundtruth[:-2, 0:500]+y_onset_pad[0:-2, 0:500]*10)
    plt.title('groundtruth onset')
    plt.show()

input_dir = 'data/error/*/'
for dir in glob.glob(input_dir):
    print(dir)
    plot(dir)