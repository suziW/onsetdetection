import tensorflow as tf 
import numpy as np 
import pymysql
import matplotlib.pyplot as plt
import time
import math
# def conv1d(x, w, stride=1):
#     x = tf.nn.conv1d(x, w, stride, 'SAME')
#     # x = tf.nn.bias_add(x, b)
#     return tf.nn.relu(x)

# window_size=4400
# X = tf.placeholder(tf.float32, [None, window_size], name='x_input')
# w = tf.Variable(initial_value=np.arange(802).reshape(-1, 1, 1), dtype=tf.float32)
# x = tf.reshape(X, shape=[-1, window_size, 1])
# conv1 = conv1d(x, w)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     p = sess.run(conv1, feed_dict={X:np.arange(2*window_size).reshape(-1, window_size)})
#     print(p.shape)
#     print(x.get_shape())
#     print(w) 


# x = tf.constant(np.arange(132))
# print(x)
# xx = tf.reshape(x, [-1, 3, 44])
# xxx = tf.transpose(xx, perm=[0, 2, 1])
# print(xxx)
# with tf.Session() as sess:
#     print(sess.run(xxx))

# def show():
#     # search:
#     sql = 'select * from new_table1'
#     try:
#         cur.execute(sql)
#         result = cur.fetchall()
#         print(result)
#     except Exception as e:
#         raise e
#     finally: 
#         pass
        # db.close()

# insert:
# sql = "insert into train(x_train, y_groundtruth, y_onset) values('{}', {}, {})"
# print('daslkfjadsl-----{}'.format(xxx[1]))
# print("da;fdka;sdf-----'{}'".format(xxx))
# # for i in range(1320):
# try:
#     cur.execute(sql.format(xxxx, xxx[1], int(x[2])))
#     db.commit()
#     print('try')
# except Exception as e:
#     db.rollback()
#     print('except')
# finally:
#     # print('---------------now {}/{}----------------'.format(i, 1320), end='\r')
#     pass
# db.close()

def get_index():
    db = pymysql.connect(host="localhost", user='suzi', password="1234",
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

def get_input_by_frame(frame):
    if type(frame)==list:
        if len(frame)>1:
            frame = tuple(frame)
            sql = 'select x from test where frame in {}'
        elif len(frame)==1:
            frame = frame[0]
            sql = 'select x from test where frame in ({})'
        else:
            print('$$$$$$$$$$ error input in get_input_by_frmae')
    elif type(frame)==int:
        sql = 'select x from test where frame in ({})'
    else:
        print('$$$$$$$$$$ error input in get_input_by_frmae')
    db = pymysql.connect(host="localhost", user="suzi", password="1234",
                    db="onset_detection",port=3306)
    cur = db.cursor()
    cur.execute(sql.format(frame))
    result = cur.fetchall()
    print(result)
    # result = result[0][0]
    return result
    
def insert(lis):
    # lis = pymysql.Binary(lis)
    print(lis)
    print(type(lis))
    db = pymysql.connect(host="localhost",user="root",
        password="1234",db="onset_detection",port=3306)
    cur = db.cursor() 
    sql = "insert into test(x) values(%s)" 
    try:
        cur.execute(sql, lis) 
        # cur.execute(sql.format(self.__x_input[1111], self.__y_input[1111]))
    except Exception as e:
        db.rollback()
        print('except', e)
    db.commit()
    db.close()

if __name__=='__main__':
    lis = list(np.arange(10))
    insert(lis)
    result = get_input_by_frame(8)
    print(result)
    print(type(result))
    # print(str(list(result, 2)))