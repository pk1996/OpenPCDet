'''
For livox annotations
'''
import numpy as np


def get_objects_from_label(label_file):
    '''
    API called to parse through label text file
    '''
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    # This line retians only car and pedestrian classes
    objects = [obj for obj in objects if obj.cls_id != -1]
    return objects


def cls_type_to_id(cls_type):
    '''
    Helper method to map class names to class id
    '''
    type_to_id = {'car': 1, 'pedestrian': 2}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(',')
        self.src = line
        self.trackingId = label[0]
        self.cls_type = label[1]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.h = float(label[7])
        self.w = float(label[6])
        self.l = float(label[5])
        self.loc = np.array((float(label[2]), float(label[3]), float(label[4])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[8])

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d
    
    def print_(self):
        '''
        To print content of class.
        For debugging puroposes
        '''
        print(self.cls_type, self.loc, self.l, self.w, self.h, self.ry)
        return