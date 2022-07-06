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
import torch
import numpy as np
from tqdm import tqdm
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import object3d_livox
import argparse

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
parser.add_argument('--split', default='test', help='set split - [test, train' )
args = parser.parse_args()
split = args.split

CLASS = ['car', 'pedestrian']

folder_name = 'testing' if split == 'test' else 'training'

# Paths to the anno and point cloud path
lidar_path = osp.join('data/livox', folder_name, 'points')
anno_path = osp.join('data/livox', folder_name, 'anno')

# Path to save the filtered anno points
new_anno_path = osp.join('data/livox', folder_name, 'anno_filtered_1')

if not os.path.exists(new_anno_path):
    os.mkdir(new_anno_path)

anno_file_list = os.listdir(anno_path)
for anno in tqdm(anno_file_list):
    # data paths
    fileName = anno.split('.')[0]
    anno_file = osp.join(anno_path, fileName + '.txt')
    lidar_file = osp.join(lidar_path, fileName + '.txt')

    # handle to write data
    f = open(os.path.join(new_anno_path, fileName + '.txt'), 'w')

    # Read data
    points = get_lidar(lidar_file)

    # Iterate over each annotation and check if contained in 
    # frontal point cloud
    src_lines = open(anno_file, 'r').readlines()
    for src_line in src_lines:

        # Ignore labels that do not correspond to
        # car and pedastrain
        class_name = src_line.split(',')[1]
        if class_name not in CLASS:
            continue
        
        anno = object3d_livox.Object3d(src_line)
        loc_lidar = anno.loc #calib.rect_to_lidar(loc)
        l, h, w = anno.l, anno.h, anno.w
        rots = anno.ry
        gt_boxes = np.hstack([loc_lidar, np.array([l, w, h, -(np.pi / 2 + rots)])]).reshape(1,-1)

        # Get points inside bbox
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)
        # Retain labels with atleast 1 point
        if np.any(point_indices):
            # Retain annotation
            f.write(src_line)