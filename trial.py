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

x1 = np.ones(342*12*3*88*3, dtype=np.int8).reshape(-1, 12, 3*88, 3)
x2 = np.ones(542*12*3*88*3, dtype=np.int8).reshape(-1, 12, 3*88, 3)
print(x1.shape, x2.shape)

x = []
x.append(x1)
x.append(x2)
x = np.concatenate(x)
print(x.shape)