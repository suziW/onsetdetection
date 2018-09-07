#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import mysql
import time
import pymysql
import os 
import pretty_midi
import tensorflow as tf 

x = np.random.rand(12).reshape(4, 3)
y = np.zeros(x.shape)
y[x>0.7] = 1
y[x<=0.1] = 0
print('x:', y)
y[1, 1] = 1
y[2, 2] = 1
print('y: ', y)

threshhole = tf.constant(0.7)

X = tf.placeholder(tf.float32, [None, 3], name='x_input')
Y = tf.placeholder(tf.float32, [None, 3], name='y_input')
greater_op = tf.cast(tf.greater(X, threshhole), tf.float32)
prediction_location = tf.where(tf.greater(Y, threshhole))
correct_op = tf.cast(tf.equal(greater_op, Y), tf.float32)
sum_op = tf.reduce_mean(correct_op, axis=1)
correct_prediction_op = tf.equal(sum_op, 1)
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {X: x, Y: y}
    greater, sum_, accuracy, location = sess.run([greater_op, sum_op, accuracy_op, prediction_location], feed_dict=feed_dict)
    print(greater)
    print(sum_)
    print(accuracy)
    print(location)