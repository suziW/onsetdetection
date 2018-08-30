import tensorflow as tf 
import numpy as np 
import pymysql
import matplotlib.pyplot as plt
import time
import math
import threading
import mysql

Y = tf.placeholder(tf.int32, [None], name='y_input')
y = np.random.randint(0, 2, 6)
print(y)
onehot = tf.one_hot(Y, 2, 1, 0)

with tf.Session() as sess:
    input_ = sess.run(Y, feed_dict={Y: y})
    print(input_)
    out = sess.run(onehot, feed_dict={Y:y})
    print(out)
    print(out.shape)