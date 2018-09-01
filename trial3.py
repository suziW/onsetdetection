#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import mysql
import time
import pymysql
import os 

dir = '/model/sht/goupi/xijingping/lasdfj.wav'
print(os.path.split(dir))
print(os.path.splitext(dir))

wav = np.arange(5)
for i,j in enumerate(wav):
    print(i, j)