#!/usr/bin/env python
# -*- coding: utf-8 -*-
import queue
import threading
import time
from trial3 import DataGen

class myThread(threading.Thread):
    def __init__(self, name, q, gen):
        super(myThread, self).__init__(name=name)
        self.q = q
        # global data
        self.gen = gen
        print(self.gen)
        self.inque = True
        self.name = name
        print('THREAD id {} started'.format(self.name))

    def run(self):
        while self.inque:
            self.q.put(next(self.gen()))
        print('THREAD id {} stoped'.format(self.name))

    def stop(self):
        self.inque = False
        print(self.q.qsize(), 'id', self.name)
        self.q.get(timeout=1)
if __name__ == '__main__':
    Thread_num = 3
    q = queue.Queue(100)
    data = DataGen()

    start = time.time()
    inq = []
    for i in range(Thread_num):
        inq.append(myThread(i, q, data.gendata))
        inq[i].start()
    last = time.time()
    for i in range(4):
        outq = q.get()
        time.sleep(0.1)
        print('=======', outq, '===', time.time()- last)
        last = time.time()

    for i in range(Thread_num):
        inq[i].stop()  
    print('============================ total time: ', time.time() - start)