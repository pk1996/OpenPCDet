import os 
import os.path as osp
from tkinter import Y
import numpy as np

CLASS_ID = {'car': 1, 'truck' : 2, 'bus' : 3, 'bimo' : 4, 'pedestrian' : 5, 'dog' : 6}

def get_label_data(label):
    label_data = label.split(',')
    obj = {}
    obj['cls_id'] = CLASS_ID[label_data[1]]
    # obj['x'] = float(label_data[2])
    # obj['y'] = float(label_data[3])
    # obj['z'] = float(label_data[4])
    obj['dim'] = np.array([float(label_data[5]), float(label_data[6]), float(label_data[7])])
    # obj['l'] = float(label_data[5])
    # obj['w'] = float(label_data[6])
    # obj['h'] = float(label_data[7])
    # obj['yaw'] = float(label_data[8])
    return obj


label_file = '/home/pkumar/OpenPCDet/data/livox/testing/anno'
labels = os.listdir(label_file)

indices = open('/home/pkumar/OpenPCDet/data/livox/ImageSets/train.txt', 'r').read().rsplit()

ANCHOR_BOX = {i:np.zeros((1,3)) for i in range(1,7)}
BOX_COUNT = {i:0 for i in range(1,7)}
for label in labels:
    label_ = label.split('.')[0]
    if label_ not in indices:
        continue

    label_data = open(osp.join(label_file, label), 'r').readlines()
    for label_l in label_data:
        label_dict = get_label_data(label_l)
        ANCHOR_BOX[label_dict['cls_id']] += label_dict['dim']
        BOX_COUNT[label_dict['cls_id']] += 1

CLASS_NAMES = CLASS_ID.keys()
print(CLASS_ID)
for i in range(1,7):
    ANCHOR_BOX[i] = ANCHOR_BOX[i]/BOX_COUNT[i]
    print(i, BOX_COUNT[i], ANCHOR_BOX[i])