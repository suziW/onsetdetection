#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import load
import numpy as np 
import time
from sklearn import preprocessing

window_size = 1320
dirpath = 'model/'
learning_rate = 0.001
epochs = 44
batch_size = 32
dropout = 0.75
print_step = 500
early_stop = {'best_loss': 100.0, 'tolerance':5, 'not_improve_cnt':0}

data = load.DataGen(dirpath, batch_size=batch_size, split=2000)
step_per_epoch = data.train_steps()
num_steps = epochs*step_per_epoch
x_train_shape, y_train_shape, positive_train = data.getinfo_train()
x_test_shape, y_test_shape, positive_test = data.getinfo_test()
print('>>>>>>>>>>>>>training data: ', data.getinfo_train())
print('>>>>>>>>>>>>>testing data: ', data.getinfo_test())
print('>>>>>>>>>>>>>num_steps: ', num_steps)

X = tf.placeholder(tf.float32, [None, x_train_shape[1]], name='x_input')
Y = tf.placeholder(tf.float32, [None, 1], name='y_input')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)

def conv1d(x, w, b, stride=1):
    x = tf.nn.conv1d(x, w, stride, 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, window_size, 1])
    x_window = tf.multiply(x, weights['ww'])
    conv1 = conv1d(x_window, weights['wc1'], biases['bc1'])
    # conv2 = conv1d(conv1, weights['wc2'], biases['bc2'])

    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc2 = tf.nn.relu(fc2)
    # fc2 = tf.nn.dropout(fc2, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out

weights = {
    'ww': tf.Variable(tf.ones([window_size, 1], dtype=tf.float32), name='ww'),
    'wc1': tf.Variable(tf.truncated_normal([110, 1, 88], dtype=tf.float32)/10, name='wc1'),
    'wc2': tf.Variable(tf.truncated_normal([110, 352, 88], dtype=tf.float32)/10, name='wc2'),
    'wd1': tf.Variable(tf.truncated_normal([window_size*88, 88], dtype=tf.float32)/10, name='wd1'),
    'wd2': tf.Variable(tf.truncated_normal([352, 88], dtype=tf.float32)/10, name='wd2'),
    'out': tf.Variable(tf.truncated_normal([88, 1], dtype=tf.float32)/10, name='wout')
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bc1'),
    'bc2': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bc2'),
    'bd1': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bd1'),
    'bd2': tf.Variable(tf.truncated_normal([88], dtype=tf.float32)/10, name='bd2'),
    'out': tf.Variable(tf.truncated_normal([1], dtype=tf.float32)/10, name='bout') 
}

with tf.name_scope('model_scope'):
    logits = conv_net(X, weights, biases, keep_prob)

prediction_op = tf.nn.sigmoid(logits) 

with tf.name_scope('loss_scope'):
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('loss_summary', loss_op)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merge_summary_op = tf.summary.merge_all()
tf.add_to_collection('pred_collection', prediction_op)
time_dict = {'start_time': time.localtime(), 'det_time': 0, 'last_time':time.time()}
test_x, test_y = data.get_test_data()
# test_x = preprocessing.MaxAbsScaler().fit_transform(test_x.T).T 
# test_x = preprocessing.MinMaxScaler().fit_transform(test_x.T).T
test_x = preprocessing.StandardScaler().fit_transform(test_x.T).T
# test_x = test_x * 10
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('model/logs', sess.graph)
    sess.graph.finalize()
    for step in range(1, num_steps+1):
        batch_x, batch_y = next(data.train_gen())
        # batch_x = batch_x * 10
        batch_x = preprocessing.StandardScaler().fit_transform(batch_x.T).T 
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if step % step_per_epoch == 0:
            test_loss = sess.run(loss_op,
                            feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
            if test_loss <= early_stop['best_loss']:
                print('{}!!!!!!!!!! test loss improved from {} to {}'.format(step/step_per_epoch, early_stop['best_loss'], test_loss))
                early_stop['not_improve_cnt'] = 0
                early_stop['best_loss'] = test_loss
                saver.save(sess, 'model/savers/{}'.format(step/step_per_epoch), global_step=step)
                print('model_saved')
            elif early_stop['not_improve_cnt'] == early_stop['tolerance']:
                print('{}!!!!!!!!!! early stop! test loss cant improve from {}'.format(step/step_per_epoch, early_stop['best_loss']))
                break
            else:
                early_stop['not_improve_cnt'] += 1
                print('{}!!!!!!!!!! test loss not improving! {} for {}'.format(step/step_per_epoch, early_stop['best_loss'], early_stop['not_improve_cnt']))
            learning_rate = learning_rate*0.9

        if step % print_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            train_loss, pred, summary = sess.run([loss_op, prediction_op, merge_summary_op], 
                                        feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})

            pred = np.round(pred).astype(np.int32)       
            time_dict['det_time'] = time.time() - time_dict['last_time']
            time_dict['last_time'] = time.time()
            time_dict['remain_time'] = (num_steps - step)/print_step * time_dict['det_time']
            print('===========================================================================================')
            print("-------------------Step: {}/{}  epoch: {}".format(step, num_steps, step/step_per_epoch))
            print('----------learning rate:', learning_rate)
            print("-------------batch Loss: {:.4f}".format(train_loss))
            print('--------------time left: {}h {}min'.format(time_dict['remain_time']//3600, (time_dict['remain_time']%3600)//60))
            print('------------ prediction:', pred[:31, 0])
            print('------------groundtruth:', batch_y[:31, 0])
            # print('---------------x_input :', batch_x)
            # print('-----------------  max :', np.max(batch_x))
            print('===========================================================================================')
            summary_writer.add_summary(summary, step)
              
    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    # print("Testing Accuracy:", \
    #     sess.run(prediction, feed_dict={X: data.get_test_data()[0],
    #                                   Y: data.get_test_data()[1],
    #                                   keep_prob: 1.0}))