import pymysql
import numpy as np
import time

def get_index():
    db = pymysql.connect(host="localhost", user="suzi", password="1234",
                    db="onset_detection",port=3306)

    cur = db.cursor()
    sql0 = 'select frame from train where y_onset=0'
    sql1 = 'select frame from train where y_onset=1'
    cur.execute(sql0)
    zero_index = cur.fetchall()
    cur.execute(sql1)
    one_index = cur.fetchall()
    db.close()
    zero_index = [i[0] for i in zero_index]
    one_index = [i[0] for i in one_index]
    return zero_index, one_index

def get_input_by_frame(frame, cur):
    # time1 = time.time()
    assert(type(frame)==list)
    if len(frame)>1:
        frame = tuple(frame)
        sql = 'select x_train, y_onset from train where frame in {}'
    elif len(frame)==1:
        frame = frame[0]
        sql = 'select x_train, y_onset from train where frame in ({})'
    else:
        print('$$$$$$$$$$ error input in get_input_by_frmae')
    cur.execute(sql.format(frame))
    result = cur.fetchall()

    x_train = [i[0] for i in result]
    y_onset = [i[1] for i in result]
    # float_list = [eval(x) for x in x_train]
    # time4 = time.time()
    # print('------------------:', time4-time3)
    str_list = [x.strip('[]').split(',') for x in x_train]
    # time5 = time.time()
    # print('------------------:', time5-time4)
    float_list = [[float(i) for i in x] for x in str_list]
    # time6 = time.time()
    # print('------------------:', time6-time5)
    return np.array(float_list), np.array(y_onset)
    # return y_onset, np.array(y_onset)

if __name__=='__main__':
    db = pymysql.connect(host="localhost", user="suzi", password="1234",
                    db="onset_detection",port=3306)
    cur = db.cursor()

    # zeros, ones = get_index()
    # print(len(zeros), len(ones))
    start = time.time()
    for i in range(500):
        print('============== {}/500'.format(i), end='\r')
        x, y = get_input_by_frame(list(np.arange(1, 257)), cur)
    # print(x.shape, y.shape)
    print('------------------:', time.time()-start)