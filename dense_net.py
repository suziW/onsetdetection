
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
# from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

def conv_layer(input, filter, kernel, stride=1, layer_name="conv", padding='SAME'):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x, stride=1):
    width = x.get_shape()[1]
    height = x.get_shape()[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, name='global_Average_Pooling') # The stride value does not matter

def Drop_out(x, keep_prob, training) :
    return tf.layers.dropout(inputs=x, rate=1-keep_prob, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='same'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=2, name='linear')



class Model_dense:
    def __init__(self, window_size, filters=44):
        self.window_size = window_size
        self.X = tf.placeholder(tf.float32, [None, self.window_size], name='x_input')
        self.Y = tf.placeholder(tf.int32, [None], name='y_input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)
        self.training = tf.placeholder(tf.bool, name='training_flag')

        self.label = tf.one_hot(self.Y, 2, 1.0, 0.0) 

        self.nb_blocks = [6, 12, 24, 16]
        self.filters = filters

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, training=self.training, name=scope+'_batchnorm1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, keep_prob=self.keep_prob, training=self.training)

            x = tf.layers.batch_normalization(x, training=self.training, name=scope+'_batchnorm2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 22], layer_name=scope+'_conv2')
            x = Drop_out(x, keep_prob=self.keep_prob, training=self.training)
            # print(x)
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, training=self.training, name=scope+'_batchnorm')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, keep_prob=self.keep_prob, training=self.training)
            x = Average_pooling(x, pool_size=[1,2], stride=2)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleneck' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleneck' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)
            return x

    def dense_net(self):
        with tf.name_scope('dense_net'):
            # input
            x = tf.reshape(self.X, shape=[-1, 1, self.window_size, 1])
            # x = tf.transpose(x, perm=[0, 1, 3, 2]) # (batch_size, 1, 440, 3)

            x = conv_layer(x, filter=2 * self.filters, kernel=[1,22], stride=2, layer_name='conv0')
            print(x)
            x = Max_Pooling(x, pool_size=[1,2], stride=2)   # windowsize/4 = 110
            print(x)

            # dense block 
            x = self.dense_block(input_x=x, nb_layers=self.nb_blocks[0], layer_name='block1')
            print(x)


            # x = self.transition_layer(x, scope='trans1')    # windowsize/8 = 55
            # x = tf.layers.batch_normalization(x, training=self.training, name='batchnorm_final1')
            # x = Relu(x)

            # print(int(x.get_shape()[2]), int(x.get_shape()[3]))
            # x = conv_layer(x, filter=352, kernel=[1, int(x.get_shape()[2])], layer_name='self_defined_final', padding='VALID')
            # x = tf.layers.batch_normalization(x, training=self.training, name='batchnorm_final2')
            # x = Relu(x)
            # print(x)
            # x = conv_layer(x, filter=88, kernel=[1, 1], layer_name='self_defined_final2')
            # x = tf.layers.batch_normalization(x, training=self.training, name='batchnorm_final3')
            # x = Relu(x)
            # print(x)
            # x = conv_layer(x, filter=2, kernel=[1, 1], layer_name='self_defined_final3')
            # x = tf.layers.batch_normalization(x, training=self.training, name='batchnorm_final4')
            # x = Relu(x)
            # print(x)
            # x = flatten(x)
            # print(x)
            # return(x)


            x = self.transition_layer(x, scope='trans1')    # windowsize/8 = 55
            print(x)

            x = self.dense_block(input_x=x, nb_layers=self.nb_blocks[1], layer_name='block2')
            x = self.transition_layer(x, scope='trans2')    # windowsize/16 = 28

            x = self.dense_block(input_x=x, nb_layers=self.nb_blocks[2], layer_name='block3')
            x = self.transition_layer(x, scope='trans3')    # windowsize/32 = 14

            x = self.dense_block(input_x=x, nb_layers=self.nb_blocks[3], layer_name='block4')

            # output
            x = tf.layers.batch_normalization(x, training=self.training, name='batchnorm_final')
            x = Relu(x)
            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = Linear(x)
            return x

if __name__=='__main__':
    inputx = np.random.rand(1320*8).reshape(8, 1320)
    inputy = np.random.randint(0, 2, 6)

    model = Model_dense(1320)
    logits = model.dense_net()

    print('==================================')
    print(inputx)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('model/logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        train_feed_dict = {
            model.X: inputx,
            model.Y: inputy,
            model.training : True,
            model.keep_prob : 0.0
        }
        logit, label = sess.run([logits, model.label], feed_dict=train_feed_dict)
        print('==================================')
        print(logit)
        print(label)