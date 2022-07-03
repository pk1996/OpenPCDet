'''
This is a pre-processing script to filter out 
annotations that do not lie even partially 
inside the frontal 3 lidars

To run the script - 
from OpenPCDet folder
python -m pcdet.datasets.livox.filter_anno

'''
import os
import os.path as osp
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse
# from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# from ...utils import object3d_livox

def get_lidar(lidar_file):
    '''
    Read point cloud (frontal lidars)
    Taken from livox_dataset.py file
    '''
    assert os.path.isfile(lidar_file), lidar_file
    points = [] # List to aggregate points per frame.
    with open(str(lidar_file)) as file:
        for line in file:
            pt_list = line.rstrip().split(',')
            points.append([[float(item) for item in pt_list]])
    points = np.concatenate(points)
    points = points.reshape(-1, 6)
    # Filter out lidar points corresponding to lidars of back-side and tele-lidar
    points = points[np.logical_or(points[:,-1] == 1 , points[:,-1] ==2, points[:,-1] == 5), :]
    points = points[:,0:3]
    return points 


parser = argparse.ArgumentParser()
# The location of training set
parser.add_argument('--split', default='test')
args = parser.parse_args()
split = args.split

folder_name = 'testing' if split == 'test' else 'training'

# Paths to the anno and point cloud path
lidar_path = osp.join('data/livox', folder_name, 'points')
anno_path = osp.join('data/livox', folder_name, 'anno')

anno_file_list = os.listdir(anno_path)
XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = (0,)*6
for anno in tqdm(anno_file_list):
    # data paths
    fileName = anno.split('.')[0]
    anno_file = osp.join(anno_path, fileName + '.txt')
    lidar_file = osp.join(lidar_path, fileName + '.txt')

    # Read data
    points = get_lidar(lidar_file)
    XMIN = min(XMIN, np.min(points[:,0]))
    XMAX = max(XMAX, np.max(points[:,0]))

    YMIN = min(YMIN, np.min(points[:,1]))
    YMAX = max(YMAX, np.max(points[:,1]))

    ZMIN = min(ZMIN, np.min(points[:,2]))
    ZMAX = max(ZMAX, np.max(points[:,2]))


print(XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)