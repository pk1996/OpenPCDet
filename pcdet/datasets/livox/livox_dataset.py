import copy
import pickle

import numpy as np
from pathlib import Path

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_livox #calibration_kitti
from ..dataset import DatasetTemplate


class LivoxDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.livox_infos = []
        self.include_livox_data(self.mode)

    def include_livox_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading LIVOX dataset')
        livox_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                livox_infos.extend(infos)

        self.livox_infos.extend(livox_infos)

        if self.logger is not None:
            self.logger.info('Total samples for LIVOX dataset: %d' % (len(livox_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        '''
        Method to get point cloud information from text file.
        Filters the back side and frontal lidar
        '''
        lidar_file = self.root_split_path / 'points' / ('%s.txt' % idx)
        assert lidar_file.exists()
        points = [] # List to aggregate points per frame.
        with open(str(lidar_file)) as file:
            for line in file:
                pt_list = line.rstrip().split(',')
                points.append([[float(item) for item in pt_list]])
        points = np.concatenate(points)
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
        return points_filt

    def get_label(self, idx):
        '''
        Uses the object 3d livox helper script to parse 
        through the annotation file to generate annotations
        '''
        # label_file = self.root_split_path / 'anno_filtered' / ('%s.txt' % idx)
        label_file = self.root_split_path / 'anno' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_livox.get_objects_from_label(label_file)

    
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 3, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                if len(obj_list) != 0:
                    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)# if len(obj_list) > 0 else []
                    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                    # annotations['score'] = np.array([obj.score for obj in obj_list])

                    num_objects = len([obj.cls_type for obj in obj_list])
                    num_gt = len(annotations['name'])
                    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                    annotations['index'] = np.array(index, dtype=np.int32)

                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                    loc_lidar = loc
                    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    # loc_lidar[:, 2] += h[:, 0] / 2 ## livox annotations gives center coordinates
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                else:
                    print('%s sample_idx: %s' % (self.split, sample_idx))
                info['annos'] = annotations

                # if count_inside_pts:
                #     points = self.get_lidar(sample_idx)
                #     calib = self.get_calib(sample_idx)
                #     pts_rect = calib.lidar_to_rect(points[:, 0:3])

                #     fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                #     pts_fov = points[fov_flag]
                #     corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                #     num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                #     for k in range(num_objects):
                #         flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                #         num_points_in_gt[k] = flag.sum()
                #     annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('livox_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        '''
        I am not sure how about the use of the gt database pkl. It saves meta data of each ibject in the point cloud.
        In terms of size it is not that big (for kitti ~150 MB). Secondly it does this for classes that are not of interest lile dog, etc. 
        '''
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            if not annos:
                continue
            names = annos['name']
            # difficulty = annos['difficulty']
            # bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                            #    'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    # TODO - Evalutation code
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    # TODO - Yet to touch evaluation part of code
    def evaluation(self, det_annos, class_names, **kwargs):
        return None, {}
        # if 'annos' not in self.kitti_infos[0].keys():
        #     return None, {}

        # from .kitti_object_eval_python import eval as kitti_eval

        # eval_det_annos = copy.deepcopy(det_annos)
        # eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        # ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        # return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.livox_infos) * self.total_epochs

        return len(self.livox_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.livox_infos)

        info = copy.deepcopy(self.livox_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            # 'calib': calib,
        }

        if 'annos' in info and bool(info['annos']):
            annos = info['annos']
            # annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        else:
            gt_names = np.empty((0,))
            gt_boxes_lidar = np.empty((0,7))
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        # data_dict['image_shape'] = img_shape
        return data_dict


def create_livox_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    '''
    Called independently as pre-processing to generate data info saved ass pkl file. 
    Uses data-specific method to generate this meta information (get_infos)
    '''
    dataset = LivoxDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('livox_infos_%s.pkl' % train_split)
    val_filename = save_path / ('livox_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'livox_infos_trainval.pkl'
    test_filename = save_path / 'livox_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_livox_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_livox_infos(
            dataset_cfg=dataset_cfg,
            class_names=['car', 'pedestrian'],
            data_path=ROOT_DIR / 'data' / 'livox',
            save_path=ROOT_DIR / 'data' / 'livox'
        )
    # import sys
    # import yaml
    # from pathlib import Path
    # from easydict import EasyDict
    # dataset_cfg = EasyDict(yaml.safe_load("tools/cfgs/dataset_configs/kitti_dataset.yaml "))
    # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    # create_livox_infos(
    #     dataset_cfg=dataset_cfg,
    #     class_names=['Car', 'Pedestrian', 'Cyclist'],
    #     data_path=ROOT_DIR / 'data' / 'kitti',
    #     save_path=ROOT_DIR / 'data' / 'kitti'
    # ) 