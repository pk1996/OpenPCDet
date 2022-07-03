'''
Simple script to generate the val text file in imageset folder
I forgot to generate this in the data_organize.py!
'''

import os
import os.path as osp
import numpy as np

base_root = '/home/pkumar/OpenPCDet/data/livox/ImageSets'
f_name_val = 'val.txt'

# Read train indexes
f_name_train = 'train.txt'
f = open(osp.join(base_root, f_name_train))
lines = f.readlines()
train_idx = [line for line in lines]

# Sample 10% reandomly
idx = sorted(np.random.choice(len(train_idx), int(0.1*len(train_idx))))

f = open(osp.join(base_root, f_name_val), 'w')
for i in range(len(idx)):
    f.write(train_idx[idx[i]])
