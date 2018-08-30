import pymysql
import numpy as np
import time

def get_index(cur):
    sql0 = 'select frame from maps where y_onset=0'
    sql1 = 'select frame from maps where y_onset=1'
    cur.execute(sql0)
    zero_index = cur.fetchall()
    cur.execute(sql1)
    one_index = cur.fetchall()
    zero_index = [i[0] for i in zero_index]
    one_index = [i[0] for i in one_index]
    return zero_index, one_index

def get_input_by_frame(frame, cur):
    frame = tuple(frame)
    sql = 'select x_train, y_onset from maps where frame in {}'
    cur.execute(sql.format(frame))
    result = cur.fetchall()
    return result

if __name__=='__main__':
    db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="onset_detection",port=3306)
    cur = db.cursor()

    zeros, ones = get_index(cur)
    print(len(zeros), len(ones))
    start = time.time()
    for i in range(500):
        print('============== {}/500'.format(i), end='\r')
        x, y = get_input_by_frame(list(np.arange(i, 256+i)), cur)
    print('\n', x.shape, y.shape)
    print('------------------:', time.time()-start)