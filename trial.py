import time
from load_db import DataGen

batch_size = 256

data = DataGen(batch_size=batch_size)
start = time.time()

for i in range(500):
    x, y = next(data.train_gen())
    print('================ {}/500 ===='.format(i), end='\r')
print('================= time', time.time() - start)