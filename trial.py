import time
from load_db import DataGen
import pymysql
import mysql
import numpy as np
from sklearn import preprocessing

batch_size = 256
db = pymysql.connect(host="localhost", user="root", password="1234",
            db="onset_detection",port=3306)
cur = db.cursor()
data = DataGen(batch_size=batch_size)
start = time.time()

for i in range(500):
    frames = next(data.val_gen())
    # print(len(frames))

    print('================ {}/500 ===='.format(i), end='\r')
print('\n ================= time', time.time() - start)