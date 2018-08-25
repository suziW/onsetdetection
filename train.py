#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf 
from load_db import DataGen
import numpy as np 
import time
from sklearn import preprocessing
from model import Model_advance, Model_base, Model_deep
import os 

window_size = 1320
dirpath = 'model/'
learning_rate = 0.001
epochs = 44
batch_size = 256
dropout = 0.75
print_step = 500
early_stop = {'best_accuracy': 0.0, 'tolerance':5, 'not_improve_cnt':0}

data = DataGen(batch_size=batch_size, split=0.99)
step_per_epoch = data.train_steps()
num_steps = epochs*step_per_epoch
print('>>>>>>>>>>>>>>>>>> train/val:len/steps: ', data.get_param())
########################################################################################################################################################################################  
########################################################################################################################################################################################  

model = Model_deep(window_size)
logits = model.deep_net()
prediction_op = tf.nn.sigmoid(logits) 

with tf.name_scope('optimize_scope'):
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=model.Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*dense_scope')
    # train_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*dense_scope|conv_weights_scope')
    # # print(train_vars1)
    # train_op1 = optimizer.minimize(loss_op, var_list=train_vars1)
    # train_op2 = optimizer.minimize(loss_op, var_list=train_vars2)
    train_op3 = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('loss_summary', loss_op)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merge_summary_op = tf.summary.merge_all()
tf.add_to_collection('pred_collection', prediction_op)
time_dict = {'start_time': time.time(), 'det_time': 0, 'last_time':time.time()}
# val_x, val_y = data.get_val_data()
# val_x = preprocessing.MaxAbsScaler().fit_transform(val_x.T).T 
# val_x = preprocessing.MinMaxScaler().fit_transform(val_x.T).T
# val_x = preprocessing.StandardScaler().fit_transform(val_x.T).T
# val_x = val_x * 10
########################################################################################################################################################################################  
########################################################################################################################################################################################  
def train_method(train_op, learning_rate=learning_rate):
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
        sess.run(train_op, feed_dict={model.X: batch_x, model.Y: batch_y, model.keep_prob: dropout})

        if step % step_per_epoch == 0:     # one epoch done, evaluate model
            equal_cnt = 0
            for _ in range(data.val_steps()):
                val_x, val_y = next(data.val_gen())
                test_pred = sess.run(prediction_op,
                                feed_dict={model.X: val_x, model.Y: val_y, model.keep_prob: 1.0})
                test_pred = np.round(test_pred).astype(np.int32)
                equal_cnt += np.sum(val_y == test_pred)
            accuracy = equal_cnt/data.get_val_len()
            print('###########################################')
            print('epoch ', step/step_per_epoch)
            print('best_accuracy: ', early_stop['best_accuracy'])
            print('test accuracy: ', accuracy)
            if accuracy >= early_stop['best_accuracy']:
                print('test accuracy improved ')
                early_stop['not_improve_cnt'] = 0
                early_stop['best_accuracy'] = accuracy
                saver.save(sess, 'model/savers/{}-{}'.format(time.time()-time_dict['start_time'], step/step_per_epoch), global_step=step)
                print('model_saved')
            elif early_stop['not_improve_cnt'] == early_stop['tolerance']:
                print('early stop! test accuracy cant improve for many epochs')
                early_stop['not_improve_cnt'] = 0
                break
            else:
                early_stop['not_improve_cnt'] += 1
                print('test accuracy not improving for {}'.format(early_stop['not_improve_cnt']))
            learning_rate = learning_rate*0.9
            print('###########################################')

        if step % print_step == 0 or step == 1:
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
            print('===========================================================================================')
            summary_writer.add_summary(summary, step)
        # break
    sess.close()


# train_method(train_op1)
# train_method(train_op2)
train_method(train_op3)              
print("Optimization Finished!")