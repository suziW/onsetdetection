import tensorflow as tf 
import numpy as np 

def conv1d(x, w, stride=1):
    x = tf.nn.conv1d(x, w, stride, 'SAME')
    # x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

window_size=4400
X = tf.placeholder(tf.float32, [None, window_size], name='x_input')
w = tf.Variable(initial_value=np.arange(802).reshape(-1, 1, 1), dtype=tf.float32)
x = tf.reshape(X, shape=[-1, window_size, 1])
conv1 = conv1d(x, w)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    p = sess.run(conv1, feed_dict={X:np.arange(2*window_size).reshape(-1, window_size)})
    print(p.shape)
    print(x.get_shape())
    print(w) 

