'''
Helpr script to divide the data in /data/livox/ into train and test split
Expected dir structure

/data/livox
  - anno 
  - points

Final dir structure
/data/livox
  -testing
     -points
     -anno

  -training
     -points
     -anno
'''

from genericpath import exists
import os
import os.path as osp
import numpy as np
import shutil
from tqdm import tqdm

def idx2name(i):
    '''
    Given idx, converts to filename
    Ex - 112 -> 000112.txt
    '''
    a_file_n = str(i)
    p_file_n = str(i)
    a_file_n = '0'*(6-len(a_file_n)) + a_file_n + '.txt'
    p_file_n = '0'*(6-len(p_file_n)) + p_file_n + '.txt'
    return a_file_n, p_file_n, p_file_n.split('.')[0]

# base folder
cwd = os.path.dirname(os.path.realpath('__file__'))
base_path = os.path.join(cwd, '../../../data/livox')
# base_path = os.path.join(cwd, 'data', 'livox')
base_path = os.path.abspath(base_path)

# path for anno and points folder
anno_file_path = osp.join(base_path , 'anno')#anno_test
anno_file_list = os.listdir(anno_file_path)
points_file_path = osp.join(base_path , 'points')#points_test
points_file_list = os.listdir(points_file_path)

# Generate idx for train, val and test split.
# Note - val idx are 10% of train data.
k = len(anno_file_list)
split = 0.7
train_idx = np.random.choice(np.arange(k), int(split*k),replace=False)
test_idx = []
for i in range(k):
    if i not in train_idx:
        test_idx.append(i)
test_idx = np.array(test_idx)
val_idx = np.random.choice(train_idx, int(0.1*train_idx.shape[0]), replace = False)
train_idx_ = [idx for idx in train_idx if idx not in val_idx]
train_idx = np.array(train_idx_)

print('Training data.....')
# Create test and training dir
test_base_path = osp.join(base_path, 'testing')
train_base_path = osp.join(base_path, 'training')
imageset_path = osp.join(base_path, 'ImageSets')
testset = osp.join(imageset_path, 'test.txt')
trainset = osp.join(imageset_path, 'train.txt')
valset = osp.join(imageset_path, 'val.txt')

if not osp.exists(test_base_path):
    os.mkdir(test_base_path)
    os.mkdir(osp.join(test_base_path,'anno'))
    os.mkdir(osp.join(test_base_path,'points'))
    os.mkdir(train_base_path)
    os.mkdir(osp.join(train_base_path,'anno'))
    os.mkdir(osp.join(train_base_path,'points'))
    os.mkdir(imageset_path)

print('Test data.....')
# Create test split
f = open(testset, 'w')
for i,idx in enumerate(tqdm(test_idx)):
    a_file_s = osp.join(anno_file_path, anno_file_list[idx])
    p_file_s = osp.join(points_file_path, points_file_list[idx])
    a_file_n, p_file_n, idx_name = idx2name(i)
    a_file_path = osp.join(test_base_path, 'anno', a_file_n)
    p_file_path = osp.join(test_base_path, 'points', p_file_n)
    f.write(idx_name + '\n')
    shutil.move(a_file_s, a_file_path)
    shutil.move(p_file_s, p_file_path)

# Create train split
f = open(trainset, 'w')
for i,idx in enumerate(tqdm(train_idx)):
    a_file_s = osp.join(anno_file_path, anno_file_list[idx])
    p_file_s = osp.join(points_file_path, points_file_list[idx])
    a_file_n, p_file_n, idx_name = idx2name(i)
    a_file_path = osp.join(train_base_path, 'anno',  a_file_n)
    p_file_path = osp.join(train_base_path, 'points', p_file_n)
    f.write(idx_name + '\n')
    shutil.move(a_file_s, a_file_path)
    shutil.move(p_file_s, p_file_path)

print('Val data.....')
# Create val imageset file
f = open(valset, 'w')
for i,idx in enumerate(tqdm(val_idx)):
    _, _, idx_name = idx2name(i)
    f.write(idx_name + '\n')


'''
Small Bug - The data points corresponding to eval is not moved 
to training/anno and training/points.
Kindly do it manually using the following - 

mv points/* training/points/
mv anno/* training/anno/


'''