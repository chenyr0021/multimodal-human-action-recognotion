import os
import numpy as np
path = './imu'
side = 0
files = [f for f in  os.listdir(path) if f[-5]==str(side)]
n = len(files)
np.random.shuffle(files)

train = files[:int(n*0.8)]
test = files[int(n*0.8):]

with open('./split/train_split_%d.txt'%side, 'w+') as f:
    for file in train:
        f.write(file + '\n')
with open('./split/test_split_%d.txt'%side, 'w+') as f:
    for file in test:
        f.write(file + '\n')
