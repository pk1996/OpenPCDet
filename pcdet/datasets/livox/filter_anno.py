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
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import object3d_livox

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


split = 'train'

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
    anno = object3d_livox.get_objects_from_label(anno_file)
    src_lines = open(anno_file, 'r').readlines()

    for i in range(len(anno)):
        loc_lidar = anno[i].loc #calib.rect_to_lidar(loc)
        l, h, w = anno[i].l, anno[i].h, anno[i].w
        # loc_lidar[2] += h / 2
        rots = anno[i].ry
        gt_boxes = np.hstack([loc_lidar, np.array([l, w, h, -(np.pi / 2 + rots)])]).reshape(1,-1)
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)
        if np.any(point_indices):
            # Retain annotation
            f.write(src_lines[i])

        # Note - This might be a potential source of error. I have not 
        # verified if the above logic is correct by manually visualizing 
        # the data and boxes!