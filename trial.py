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

a = np.zeros(9).reshape(3, 3)
b = np.zeros(9).reshape(3, 3)

b[2, 1] = 1
b[2, 2] = 1
a[1, 2] = 1
a[2, 2] = 1
print(a)
print(b)

c = (a+b == 2)
# c[a.any()==1 and b.any()==1] = 1
# print(c)
print(a+b==2)
print(c!=a)

print(np.where(b+a>=1))