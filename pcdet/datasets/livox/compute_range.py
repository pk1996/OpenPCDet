'''
Script to compute the range of data after retaining 
only frontal lidars!
'''

import numpy as np
import os
from tqdm import tqdm
import json

base_path = '/home/pkumar/OpenPCDet/data/livox/training/points'
lidar_files = os.listdir(base_path)

XMIN = 1000000
XMAX = -1000000
YMIN = 1000000
YMAX = -1000000
ZMIN = 1000000
ZMAX = -1000000

for lidar_file in tqdm(lidar_files):
    lidar_file_path = os.path.join(base_path, lidar_file)
    points = [] # List to aggregate points per frame.
    with open(str(lidar_file_path)) as file:
        for line in file:
            pt_list = line.rstrip().split(',')
            points.append([[float(item) for item in pt_list]])
    points = np.concatenate(points).astype(np.float32)
    points = points.reshape(-1, 6)
    
    # Extract points corresponding to the frontal lidar
    # Lidar Id - 1,2,5 (frontal) | 6 - tele-lidar | 4,3 (back)
    lidar_number = [1,2,5] 
    points_filt = []
    for ln in lidar_number:
        points_filt.append(points[points[:,-1] == ln, :])
    points_filt = np.concatenate(points_filt)
    # Extract only xyz infomration
    points_filt = points_filt[:,[0,1,2]]

    # compute ranges
    XMIN = min(XMIN, np.min(points_filt[:,0]))
    XMAX = max(XMAX, np.max(points_filt[:,0]))

    YMIN = min(YMIN, np.min(points_filt[:,1]))
    YMAX = max(YMAX, np.max(points_filt[:,1]))

    ZMIN = min(ZMIN, np.min(points_filt[:,2]))
    ZMAX = max(ZMAX, np.max(points_filt[:,2]))

# [minx, miny, minz, maxx, maxy, maxz]
range = {}
range['XMIN'] = str(XMIN)
range['XMAX'] = str(XMAX)
range['YMIN'] = str(YMIN)
range['YMAX'] = str(YMAX)
range['ZMIN'] = str(ZMIN)
range['ZMAX'] = str(ZMAX)

print(XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

json.dump(range, open('range.json', 'w'))
