#!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
# import pretty_midi
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import tensorflow as tf 
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
    
sr = 22050
midinote = 72
frequnces = librosa.midi_to_hz(range(21, 109))
window_size = 440
note_range = 88
units = [88, 88, 1, 88, 1]


class Model_advance:
    def __init__(self, window_size=window_size):
        self.window_size = window_size
        self.X = tf.placeholder(tf.float32, [None, self.window_size], name='x_input')
        self.Y = tf.placeholder(tf.float32, [None, 1], name='y_input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)
        self.conv_weights = []
        self.window_weights = []
        self.init_weights()

    def sin_gen(self, frequnce, sr=22050):
        nn = sr/frequnce
        x = np.arange(nn)/sr
        y = 0.1*np.sin(2*np.pi*frequnce*x)
        # print(frequnce, x.shape, y.shape)
        return y.reshape(-1, 1, 1)

    def conv1d(self, x, w, stride=1):
        x = tf.nn.conv1d(x, w, stride, 'SAME')
        # x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def dense(self, input, units, activation=None):
        with tf.name_scope('dense_scope'):
            input_dim = int(input.get_shape()[1])
            w = tf.Variable(tf.truncated_normal([input_dim, units], dtype=tf.float32)/10, name='dense_weight')
            b = tf.Variable(tf.zeros([units], dtype=tf.float32), name='dense_biase')
            dense_mid = tf.add(tf.matmul(input, w), b)
            if activation == None:
                return dense_mid
            else:
                return activation(dense_mid)
            
    def stream_net(self, x, conv_weight, window_weight):
        with tf.name_scope('stream_scope'):
            x_window = tf.multiply(x, window_weight)
            conv1 = self.conv1d(x_window, conv_weight)
            flatten = tf.layers.flatten(conv1)
            # fc1 = self.dense(flatten, units[0], activation=tf.nn.relu)
            # fc1 = tf.layers.dropout(fc1, self.keep_prob)
            fc2 = self.dense(flatten, units[1], activation=tf.nn.relu)
            fc2 = tf.layers.dropout(fc2, self.keep_prob)
            fc3 = self.dense(fc2, units[2], activation=tf.nn.relu) 
            return fc3  

    def merge_net(self):
        x = tf.reshape(self.X, shape=[-1, self.window_size, 1])
        with tf.name_scope('multi_stream_scope'):
            stream = []
            for i in range(note_range):
                stream.append(self.stream_net(x, self.conv_weights[i], self.window_weights[i]))
        merge_layer = tf.concat(values=stream, axis=1)
        # hidden = self.dense(merge_layer, units[3], activation=tf.nn.relu)
        # hidden = tf.layers.dropout(hidden, self.keep_prob)
        out = self.dense(merge_layer, units[4])
        return out

    def init_weights(self):
        with tf.name_scope('conv_weights_scope'):
            for i in range(note_range):
                conv_weight = tf.Variable(initial_value=self.sin_gen(frequnces[i]), dtype=tf.float32, name='conv_weight{}'.format(i))
                self.conv_weights.append(conv_weight)
        with tf.name_scope('window_weights_scope'):
            for i in range(note_range):
                window_weight = tf.Variable(tf.ones([self.window_size, 1]), name='window_weight{}'.format(i))
                self.window_weights.append(window_weight)

class Model_base:
    def __init__(self, window_size):
        self.window_size = window_size
        self.X = tf.placeholder(tf.float32, [None, self.window_size], name='x_input')
        self.Y = tf.placeholder(tf.float32, [None, 1], name='y_input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)
        with tf.name_scope('panram_scope'):
            self.weights = {
                'ww': tf.Variable(tf.ones([self.window_size, 1], dtype=tf.float32), name='ww'),
                'wc1': tf.Variable(tf.truncated_normal([110, 1, 88], dtype=tf.float32)/10, name='wc1'),
                'wc2': tf.Variable(tf.truncated_normal([110, 352, 88], dtype=tf.float32)/10, name='wc2'),
                'wd1': tf.Variable(tf.truncated_normal([self.window_size*88, 88], dtype=tf.float32)/10, name='wd1'),
                'wd2': tf.Variable(tf.truncated_normal([352, 88], dtype=tf.float32)/10, name='wd2'),
                'out': tf.Variable(tf.truncated_normal([88, 1], dtype=tf.float32)/10, name='wout')
            }

            self.biases = {
                'bc1': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bc1'),
                'bc2': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bc2'),
                'bd1': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bd1'),
                'bd2': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bd2'),
                'out': tf.Variable(tf.truncated_normal([1], dtype=tf.float32)/10, name='bout') 
            }

    def conv1d(self, x, w, b, stride=1):
        x = tf.nn.conv1d(x, w, stride, 'SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def conv_net(self):
        with tf.name_scope('convnet_scope'):
            x = tf.reshape(self.X, shape=[-1, self.window_size, 1])
            x_window = tf.multiply(x, self.weights['ww'])
            conv1 = self.conv1d(x_window, self.weights['wc1'], self.biases['bc1'])
            # conv2 = conv1d(conv1, self.weights['wc2'], self.biases['bc2'])

            fc1 = tf.reshape(conv1, [-1, self.weights['wd1'].get_shape()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

            # fc2 = tf.add(tf.matmul(fc1, self.weights['wd2']), self.biases['bd2'])
            # fc2 = tf.nn.relu(fc2)
            # fc2 = tf.nn.dropout(fc2, self.keep_prob)

            out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
            
            return out


if __name__=='__main__':
    model = Model_advance()
    out_op = model.merge_net()
    tf.summary.histogram('out', out_op)
    merge_summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter('model/logs', tf.get_default_graph())
        summary, _ = sess.run([merge_summary_op, out_op], feed_dict={model.X:np.arange(2*window_size).reshape(-1, window_size),model.Y:[[0],[1]], model.keep_prob:1})
        summary_writer.add_summary(summary)
        print('done')