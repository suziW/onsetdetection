import pymysql
import numpy as np
import time

def get_index(cur):
    sql = 'select count(*) from maps'
    cur.execute(sql)
    index = cur.fetchall()[0][0]
    index = np.arange(index) + 1
    return index

def get_input_by_frame(frame, cur):
    frame = tuple(frame)
    sql = 'select x_train, y_onset from maps where frame in {}'
    cur.execute(sql.format(frame))
    result = cur.fetchall()
    return result

if __name__=='__main__':
        db = pymysql.connect(host="localhost", user="root", password="1234",
                    db="polyphonic",port=3306)
        cur = db.cursor()
        a = get_index(cur)
        print(a.shape)
        print(a[0])