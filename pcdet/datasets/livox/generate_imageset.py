'''
Helper script to generate the Imageset text file
'''
import os
import os.path as osp
import numpy as np

cwd = os.path.dirname(os.path.realpath('__file__'))
base_path = os.path.join(cwd, '../../../data/livox')
base_path = os.path.abspath(base_path)
imageset_path = osp.join(base_path, 'ImageSets1')
if not osp.exists(imageset_path):
    os.mkdir(imageset_path)

# test imageset
testset = osp.join(imageset_path, 'test.txt')
anno_path = osp.join(base_path, 'testing', 'anno')
file_list = os.listdir(anno_path)
f = open(testset, 'w')

for file_name in file_list:
    name = file_name.split('.')[0]
    f.write(name + '\n')

# train imageset
trainset = osp.join(imageset_path, 'train.txt')
valset = osp.join(imageset_path, 'val.txt')
anno_path = osp.join(base_path, 'training', 'anno')
file_list = os.listdir(anno_path)

# Randomly select 10% train data points as val
train_idx = np.arange(len(file_list))
val_idx = np.random.choice(train_idx, int(0.1*train_idx.shape[0]), replace = False)

f_train = open(trainset, 'w')
f_val = open(valset, 'w')

for i, file_name in enumerate(file_list):
    name = file_name.split('.')[0]
    if i in val_idx:
        f_val.write(name + '\n')
    else:
        f_train.write(name + '\n')