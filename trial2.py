import tensorflow as tf 
import numpy as np 
import pymysql
import matplotlib.pyplot as plt
import time
import math
import threading
import mysql

batch_size = 256
db = pymysql.connect(host="localhost", user="root", password="1234",
            db="onset_detection",port=3306)
cur = db.cursor()



 # 声明一个先进先出的队列，队列中最多100个元素，类型为实数
queue = tf.RandomShuffleQueue(50, 10, dtypes=tf.int32, shapes=[])
# 定义队列的入队操作
enqueue_op = queue.enqueue_many(np.arange(100))

# 使用 tf.train.QueueRunner来创建多个线程运行队列的入队操作
# tf.train.QueueRunner给出了被操作的队列，[enqueue_op] * 5
# 表示了需要启动5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
# 将定义过的QueueRunner加入TensorFlow计算图上指定的集合
# tf.train.add_queue_runner函数没有指定集合，
# 则加入默认集合tf.GraphKeys.QUEUE_RUNNERS。
# 下面的函数就是将刚刚定义的qr加入默认的tf.GraphKeys.QUEUE_RUNNERS结合
tf.train.add_queue_runner(qr)
# 定义出队操作
outqueue = queue.dequeue_many(3)
x = mysql.get_input_by_frame(outqueue, cur)
print('??', outqueue)


with tf.Session() as sess:
    start = time.time()
    # 使用tf.train.Coordinator来协同启动的线程
    coord = tf.train.Coordinator()
    # 使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners
    # 来启动所有线程。否则因为没有线程运行入队操作，当调用出队操作时，程序一直等待
    # 入队操作被运行。tf.train.start_queue_runners函数会默认启动
    # tf.GraphKeys.QUEUE_RUNNERS中所有QueueRunner.因为这个函数只支持启动指定集合中的QueueRunner,
    # 所以一般来说tf.train.add_queue_runner函数和tf.train.start_queue_runners函数会指定同一个结合
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for i in range(300): 
        out_data = sess.run(x)
        print('----------------- {}/500------------'.format(i), out_data)
    print('\n', '-----------------time: ', time.time() - start)
    # 使用tf.train.Coordinator来停止所有线程
    coord.request_stop()
    coord.join(threads)
    print(out_data)