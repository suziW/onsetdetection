#!/usr/bin/env python
# -*- coding: utf-8 -*-
import queue
import threading
import time
from trial3 import DataGen

data = DataGen()
Thread_num = 15
class myThread(threading.Thread):
    def __init__(self, name, q):
        super(myThread, self).__init__(name=name)
        global data
        self.q = q
        self.gen = data.gendata()
        self.stop = False
        self.name = name
        print('THREAD id {} started'.format(self.name))

    def run(self):
        while not self.stop:
            self.q.put(next(self.gen))
        print('THREAD id {} stoped'.format(self.name))

    def stopq(self):
        self.stop = True
        print(q.qsize(), 'id', self.name)
        self.q.get(timeout=1)

q = queue.Queue(100)

#向资源池里面放10个数用作测试

#开Thread_num个线程 
start = time.time()
inq = []
for i in range(Thread_num):
    inq.append(myThread(i, q))
    inq[i].start()
last = time.time()
for i in range(20):
    outq = q.get()
    # time.sleep(0.1)
    print('=======', outq, '===', time.time()- last)
    last = time.time()

for i in range(Thread_num):
    inq[i].stopq()  
print('============================ total time: ', time.time() - start)