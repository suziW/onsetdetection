#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import mysql
import time
import pymysql

sr = 22050
step = 0.33        # times of window_size
window_size = 60       # ms

class DataGen:
    def __init__(self):
        print('>>>>>>>>>>>>>>>>>> getting data......')       
        self.__db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="onset_detection",port=3306)
        self.__cur = self.__db.cursor()
        self.__train = list(np.arange(10))
        self.__index = 0

    def gendata(self):
        while True:
            time.sleep(1)
            x = self.__train[self.__index]
            if (self.__index+1) == len(self.__train):
                self.__index = 0
            else:
                self.__index += 1
            yield x

if __name__ == '__main__':
    data = DataGen()
    for i in range(13):
        print(next(data.gendata()))