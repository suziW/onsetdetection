#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql
from sklearn import preprocessing
import mysql
import queue
import threading
import time
from frames_generater import DataGen
import numpy as np

lock=threading.Lock()   #全局的锁对象

class myThread(threading.Thread):
    def __init__(self, name, q, gen):
        super(myThread, self).__init__(name=name)
        global lock

        self.__db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="onset_detection",port=3306)
        self.__cur = self.__db.cursor()

        self.q = q
        self.gen = gen
        # print(self.gen)
        # print(next(self.gen()))
        self.enque = True
        self.name = name
        self.sql = 'select x_train, y_onset from maps_final where frame in {}'
        print('THREAD id {} started'.format(self.name))

    def run(self):
        while self.enque:
            # print(self.q.qsize(), 'THREAD id {}'.format(self.name))
            frames = tuple(next(self.gen()))
            self.__cur.execute(self.sql.format(frames))
            result = self.__cur.fetchall()
            x_train = [np.fromstring(i[0], dtype=np.float32) for i in result]
            y_onset = [i[1] for i in result]
            # float_list = [np.fromstring(x, dtype=np.float32) for x in x_train]
            # for i, j in enumerate(float_list): 
            #     if j = []:
            #         y_onset.remove()
            x, y = np.array(x_train), np.array(y_onset)
            x = preprocessing.StandardScaler().fit_transform(x.T).T 
            self.q.put([x, y])

        print('THREAD id {} stoped'.format(self.name))

    def stop(self):
        self.enque = False
        print(self.q.qsize(), 'id', self.name)
        self.q.get(timeout=1)

class InputGen:
    def __init__(self, batch_size=256, split=0.99, thread_num=9):
        print('>>>>>>>>>>>>>>>>>> init InputGen......')
        self.thread_num = thread_num
        self.data = DataGen(batch_size=batch_size, split=split)
        self.train_queue = queue.Queue(500)
        self.val_queue = queue.Queue(200)

        self.train_enqueue_thread = []
        for i in range(self.thread_num):
            self.train_enqueue_thread.append(myThread('train-{}'.format(i), self.train_queue, self.data.train_gen))
            self.train_enqueue_thread[i].start()

        self.val_enqueue_thread1 = myThread('val-1', self.val_queue, self.data.val_gen)
        self.val_enqueue_thread2 = myThread('val-2', self.val_queue, self.data.val_gen)
        self.val_enqueue_thread1.start()
        self.val_enqueue_thread2.start()
        print('>>>>>>>>>>>>>>>>>> init InputGen done.')

    def val_gen(self):
        # print('>>>>>>>>>>>>>>>>>> val queue:', self.val_queue.qsize())
        yield self.val_queue.get()
    def train_gen(self):
        # print('>>>>>>>>>>>>>>>>>> train queue:', self.train_queue.qsize())
        yield self.train_queue.get()
    def stop(self):
        self.val_enqueue_thread1.stop()
        self.val_enqueue_thread2.stop()
        for i in range(self.thread_num):
            self.train_enqueue_thread[i].stop()
    def train_steps(self):
        return self.data.train_steps()
    def val_steps(self):
        return self.data.val_steps()
    def get_param(self):
        return self.data.get_param()
    def get_val_len(self):
        return self.data.get_val_len()

if __name__=='__main__':
    queuegen = InputGen(batch_size=256, thread_num=1)
    start = time.time()
    for i in range(500):
        data = next(queuegen.val_gen())
        print('====================================== {}/500'.format(i), end='\r')
    print('\n time', time.time() - start)
    mid = time.time()
    for i in range(500):
        data = next(queuegen.train_gen())
        print('====================================== {}/500'.format(i), end='\r')    
    print('\n time', time.time() - mid)
    print(data[1].shape, data[0].shape)
    # print(data)
    queuegen.stop()