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


class myThread(threading.Thread):
    def __init__(self, name, q, gen):
        super(myThread, self).__init__(name=name)
        self.__db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="polyphonic",port=3306)
        self.__cur = self.__db.cursor()

        self.q = q
        self.gen = gen
        # print(self.gen)
        # print(next(self.gen()))
        self.enque = True
        self.name = name
        self.sql = 'select x_input, y_input from maps where frame in {}'
        print('THREAD id {} started'.format(self.name))

    def run(self):
        while self.enque:
            # print(self.q.qsize(), 'THREAD id {}'.format(self.name))
            frames = tuple(next(self.gen()))
            self.__cur.execute(self.sql.format(frames))
            result = self.__cur.fetchall()
            x_train = [np.fromstring(i[0], dtype=np.float32) for i in result]
            y_onset = [np.fromstring(i[1], dtype=np.int8) for i in result]

            x, y = np.array(x_train), np.array(y_onset)
            x = preprocessing.StandardScaler().fit_transform(x.T).T
            self.q.put([x, y])

        print('THREAD id {} stoped'.format(self.name))

    def stop(self):
        self.enque = False
        print(self.q.qsize(), 'id', self.name)
        self.q.get(timeout=1)

class InputGen:
    def __init__(self, batch_size=256, timesteps=20, split=0.99, thread_num=9):
        print('>>>>>>>>>>>>>>>>>> init InputGen......')
        self.thread_num = thread_num
        self.data = DataGen(batch_size=batch_size, timesteps=timesteps, split=split)
        self.train_queue = queue.Queue(500)
        self.val_queue = queue.Queue(200)

        self.train_enqueue_thread = []
        for i in range(self.thread_num):
            self.train_enqueue_thread.append(myThread('train-{}'.format(i), self.train_queue, self.data.train_gen))
            self.train_enqueue_thread[i].start()

        self.val_enqueue_thread = myThread('val', self.val_queue, self.data.val_gen)
        self.val_enqueue_thread.start()
        print('>>>>>>>>>>>>>>>>>> init InputGen done.')

    def val_gen(self):
        # print('>>>>>>>>>>>>>>>>>> val queue:', self.val_queue.qsize())
        yield self.val_queue.get()
    def train_gen(self):
        # print('>>>>>>>>>>>>>>>>>> train queue:', self.train_queue.qsize())
        yield self.train_queue.get()
    def stop(self):
        self.val_enqueue_thread.stop()
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
    timesteps = 20
    queuegen = InputGen(batch_size=32, timesteps=timesteps, thread_num=1)
    print(queuegen.get_param())
    start = time.time()
    for i in range(6):
        data = next(queuegen.val_gen())
        print(data[1].shape, data[0].shape)
        y = data[1].reshape(-1, timesteps, 88)
        print(y.shape)
        for j in y[0]:
            print(np.where(j>0))
        print('====================================== {}/500'.format(i), end='\n')
    # print('\n time', time.time() - start)
    # mid = time.time()
    # for i in range(500):
    #     data = next(queuegen.train_gen())
    #     print('====================================== {}/500'.format(i), end='\r')    
    # print('\n time', time.time() - mid)

    queuegen.stop()