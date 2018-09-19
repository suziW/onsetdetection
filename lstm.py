import numpy as np
import tensorflow as tf 
from tensorflow.contrib import rnn

class Model_lstm:
    def __init__(self, window_size):
        self.window_size = window_size
        self.X = tf.placeholder(tf.float32, [None, self.window_size], name='x_input')
        self.Y = tf.placeholder(tf.float32, [None, 88], name='y_input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)
        self.training = tf.placeholder(tf.bool, name='training_flag')

        self.num_hidden = 352
        self.num_classes = 88
        self.weights, self.biases = self.init_param()

        self.kernal_sizes = [110, 110, 110]
        self.filters = [44, 88, 88]
        self.units = [352, 88]

    def conv1d(self, input, kernal_size, filters, stride=1, padding='SAME'):
        with tf.name_scope('convpool_scope'):
            input_dim = int(input.get_shape()[2])
            w = tf.Variable(tf.truncated_normal([kernal_size, input_dim, filters], dtype=tf.float32), name='conv_weight')
            b = tf.Variable(tf.zeros([filters], dtype=tf.float32), name='conv_biase')
            conv_mid = tf.nn.bias_add(tf.nn.conv1d(input, w, stride, padding), b)
            norm = tf.layers.batch_normalization(conv_mid, training=self.training)
            conv = tf.nn.relu(norm)
            return tf.layers.max_pooling1d(conv, pool_size=2, strides=2, padding=padding)

    def dense(self, input, units, activation=None):
        with tf.name_scope('dense_scope'):
            input_dim = int(input.get_shape()[1])
            w = tf.Variable(tf.truncated_normal([input_dim, units], dtype=tf.float32), name='dense_weight')
            b = tf.Variable(tf.zeros([units], dtype=tf.float32), name='dense_biase')
            dense_mid = tf.add(tf.matmul(input, w), b)
            if activation == None:
                return dense_mid
            else:
                norm = tf.layers.batch_normalization(dense_mid, training=self.training)
                return activation(norm)

    def deep_net(self):
        with tf.name_scope('deep_net'):
            x = tf.reshape(self.X, shape=[-1, self.window_size, 1])
            conv1 = self.conv1d(x, self.kernal_sizes[0], self.filters[0], stride=1) # kernal: (44, 3, 32) maxpool: 2
            conv2 = self.conv1d(conv1, self.kernal_sizes[1], self.filters[1], stride=1) # kernal: (11, 32, 64) maxpool:2
            conv3 = self.conv1d(conv2, self.kernal_sizes[2], self.filters[2], stride=1) # kernal: (4, 64, 64) maxpool:2
            flatten = tf.layers.flatten(conv3)
            fc1 = self.dense(flatten, self.units[0], activation=tf.nn.relu)
            fc2 = self.dense(fc1, self.units[1])
            return fc2

    def lstm(self):
        with tf.name_scope('lstm_scope'):
            x = tf.reshape(self.X, shape=[-1, self.window_size, 1])
            x = tf.unstack(x, self.window_size, 1)
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def init_param(self): 
        weights = {
            'out': tf.Variable(tf.truncated_normal([self.num_hidden, self.num_classes], dtype=tf.float32, name='w-out'))
        }
        biases = {
            'out': tf.Variable(tf.truncated_normal([self.num_classes], dtype=tf.float32, name='b-out'))
        }
        return weights, biases

if __name__=='__main__':
    model = Model_lstm(12)
    logits = model.lstm()

    x = np.random.rand(12*3).reshape(3, 12)
    print(x)

    tf.summary.histogram('out', logits)
    merge_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('model/logs', tf.get_default_graph())
        feed_dict = {model.X: x}
        out, summary = sess.run([logits, merge_summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary)
        print(out)