#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import load
import numpy as np 
import time
from sklearn import preprocessing
from model import Model_advance, Model_base
import os 

window_size = 1320
dirpath = 'model/'
learning_rate = 0.001
epochs = 44
batch_size = 256
dropout = 0.75
print_step = 500
early_stop = {'best_loss': 100.0, 'tolerance':3, 'not_improve_cnt':0}

data = load.DataGen(dirpath, batch_size=batch_size, split=3000)
step_per_epoch = data.train_steps()
num_steps = epochs*step_per_epoch
x_train_shape, y_train_shape, positive_train = data.getinfo_train()
x_test_shape, y_test_shape, positive_test = data.getinfo_test()
print('>>>>>>>>>>>>>training data: ', data.getinfo_train())
print('>>>>>>>>>>>>>testing data: ', data.getinfo_test())
print('>>>>>>>>>>>>>num_steps: ', num_steps)
########################################################################################################################################################################################  
########################################################################################################################################################################################  

model = Model_advance(window_size)
logits = model.merge_net()
prediction_op = tf.nn.sigmoid(logits) 

with tf.name_scope('optimize_scope'):
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=model.Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*dense_scope')
    train_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*dense_scope|conv_weights_scope')
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    # print(train_vars1)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    train_op1 = optimizer.minimize(loss_op, var_list=train_vars1)
    train_op2 = optimizer.minimize(loss_op, var_list=train_vars2)
    train_op3 = optimizer.minimize(loss_op)

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
########################################################################################################################################################################################  
########################################################################################################################################################################################  
def train_method(train_op, num, learning_rate=learning_rate):
    sess = tf.Session()
    if os.path.exists('model/savers'):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ restore model')
        saver.restore(sess, tf.train.latest_checkpoint('model/savers/'))
    else:
        sess.run(init)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ init model')
    summary_writer = tf.summary.FileWriter('model/logs', sess.graph)
    sess.graph.finalize()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ over preparing for train')
    for step in range(1, num_steps+1):
        batch_x, batch_y = next(data.train_gen())
        # batch_x = batch_x * 10
        batch_x = preprocessing.StandardScaler().fit_transform(batch_x.T).T 
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={model.X: batch_x, model.Y: batch_y, model.keep_prob: dropout})
        if step % step_per_epoch == 0:
            test_loss = sess.run(loss_op,
                            feed_dict={model.X: test_x, model.Y: test_y, model.keep_prob: 1.0})
            print('###########################################')
            print('epoch ', step/step_per_epoch)
            print('best loss: ', early_stop['best_loss'])
            print('test loss: ', test_loss)
            if test_loss <= early_stop['best_loss']:
                print('test loss improved ')
                early_stop['not_improve_cnt'] = 0
                early_stop['best_loss'] = test_loss
                saver.save(sess, 'model/savers/{}-{}'.format(num, step/step_per_epoch), global_step=step)
                print('model_saved')
            elif early_stop['not_improve_cnt'] == early_stop['tolerance']:
                print('early stop! test loss cant improve for many epochs')
                early_stop['not_improve_cnt'] = 0
                break
            else:
                early_stop['not_improve_cnt'] += 1
                print('test loss not improving for {}'.format(early_stop['not_improve_cnt']))
            learning_rate = learning_rate*0.9
            print('###########################################')
        if step % print_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            train_loss, pred, summary = sess.run([loss_op, prediction_op, merge_summary_op], 
                                        feed_dict={model.X: batch_x, model.Y: batch_y, model.keep_prob: 1.0})

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
        # break
    sess.close()


train_method(train_op1, num=1)
train_method(train_op2, num=2)
train_method(train_op3, num=3)              
print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    # print("Testing Accuracy:", \
    #     sess.run(prediction, feed_dict={X: data.get_test_data()[0],
    #                                   Y: data.get_test_data()[1],
    #                                   keep_prob: 1.0}))