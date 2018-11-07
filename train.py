#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf 
from input_queue import InputGen
import numpy as np 
import time
from sklearn import preprocessing
from model import Model_advance, Model_base, Model_deep
from dense_net import Model_dense

window_size = 1320
dirpath = 'model/'
learning_rate = 0.01
learning_decay = 0.8
epochs = 50
batch_size = 64
keep_prob = 1
print_step = 1000
early_stop = {'best_accuracy': 0.0, 'tolerance': 4, 'not_improve_cnt':0}
model_save_path = 'savers'

data = InputGen(batch_size=batch_size, split=0.99, thread_num=5)
step_per_epoch = data.train_steps()
num_steps = epochs*step_per_epoch
print('>>>>>>>>>>>>>>>>>> train/val:len/steps: ', data.get_param())
########################################################################################################################################################################################
########################################################################################################################################################################################

model = Model_dense(window_size)
logits = model.dense_net()

with tf.name_scope('optimize_scope'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model.label, logits=logits))
    # loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=model.Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(0.001, 0.4)
    # train_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*dense_scope')
    # train_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*dense_scope|conv_weights_scope')
    # # print(train_vars1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): #保证train_op在update_ops执行之后再执行.
        # train_op1 = optimizer.minimize(loss_op, var_list=train_vars1)
        # train_op2 = optimizer.minimize(loss_op, var_list=train_vars2)
        train_op3 = optimizer.minimize(loss_op)

with tf.name_scope('accuracy_scope'):
    prediction_op = tf.nn.softmax(logits)
    prediction_type = tf.argmax(prediction_op, 1)
    correct_prediction = tf.equal(tf.argmax(prediction_op, 1),tf.argmax(model.label,1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
tf.summary.scalar('loss_summary', loss_op)
tf.summary.scalar('accuracy_summary', accuracy_op)
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
    if os.path.exists('model/{}'.format(model_save_path)):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ restore model')
        saver.restore(sess, tf.train.latest_checkpoint('model/{}/'.format(model_save_path)))
    else:
        sess.run(init)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ init model')
    summary_writer = tf.summary.FileWriter('model/logs', sess.graph)
    sess.graph.finalize()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ preparing for train')
    for step in range(1, num_steps+1):
        batch_x, batch_y = next(data.train_gen())
        sess.run(train_op, feed_dict={model.X: batch_x, model.Y: batch_y, model.keep_prob: keep_prob, model.training: True})

        if step % step_per_epoch == 0:     # one epoch done, evaluate model
            print('###########################################')
            accuracy = 0
            for val_step in range(data.val_steps()):
                print('valing.....................................{}/{}.......'.format(val_step, data.val_steps()), end='\r')
                val_x, val_y = next(data.val_gen())
                test_acc, test_prediction = sess.run([accuracy_op, prediction_type], 
                                feed_dict={model.X: val_x, model.Y: val_y, model.keep_prob: 1.0, model.training: False})
                accuracy += test_acc
                # print('---------------test_acc:', test_acc)
                # print('--------test_prediction:', test_prediction[:31])
                # print('------------groundtruth:', val_y[:31])

            accuracy = accuracy/data.val_steps()
            print('valing done')
            print('epoch ', step/step_per_epoch)
            print('best_accuracy: ', early_stop['best_accuracy'])
            print('test accuracy: ', accuracy)
            if accuracy >= early_stop['best_accuracy']:
                print('test accuracy improved ')
                early_stop['not_improve_cnt'] = 0
                early_stop['best_accuracy'] = accuracy
                saver.save(sess, 'model/{}/{}-{}'.format(model_save_path, accuracy, step/step_per_epoch), global_step=step)
                print('model_saved')
            elif early_stop['not_improve_cnt'] == early_stop['tolerance']:
                print('early stop! test accuracy cant improve for many epochs')
                early_stop['not_improve_cnt'] = 0
                data.stop()
                break
            else:
                early_stop['not_improve_cnt'] += 1
                print('test accuracy not improving for {}'.format(early_stop['not_improve_cnt']))
            learning_rate = learning_rate*learning_decay
            print('###########################################')

        if step % print_step == 0 or step == 1:
            train_acc, train_loss, pred, summary = sess.run([accuracy_op, loss_op, prediction_type, merge_summary_op],
                                        feed_dict={model.X: batch_x, model.Y: batch_y, model.keep_prob: 1.0, model.training: False})
            time_dict['det_time'] = time.time() - time_dict['last_time']
            time_dict['last_time'] = time.time()
            time_dict['remain_time'] = (num_steps - step)/print_step * time_dict['det_time']
            print('===========================================================================================')
            print("-------------------Step: {}/{}  epoch: {}".format(step, num_steps, step/step_per_epoch))
            print('----------learning rate:', learning_rate)
            print("-------------batch Loss: {:.4f}".format(train_loss))
            print("---------------accuracy: {:.4f}".format(train_acc))
            print('--------------time left: {}h {}min'.format(time_dict['remain_time']//3600, (time_dict['remain_time']%3600)//60))
            print('------------ prediction:', pred[:31])
            print('------------groundtruth:', batch_y[:31])
            print('===========================================================================================')
            summary_writer.add_summary(summary, step)
        # break
    sess.close()


# train_method(train_op1)
# train_method(train_op2)
train_method(train_op3)              
print("Optimization Finished!")
print("dense")