import tempfile
from os import path as osp
import mmcv
import numpy as np
import pyquaternion

class CarlaDataset():
    OD_CLASSES_RAW = ('vehicle', 'pedestrian', 'building', 'fence', 'pole', 'road',
                        'sidewalk', 'traffic_sign', 'traffic_light', 'tree', 'wall', 'sky',
                        'ground', 'bridge', 'rail_track', 'guard_rail', 'water', 'terrain',
                        'static', 'dynamic')


    # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
    SEG_CLASSES_RAW = ('unlabeled', 'roads', 'sidewalks', 'building', 'wall', 'fence', 'pole',
                'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky', 'pedestrian', 'rider',
                'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'static', 'dynamic', 'other',
                'water', 'roadline', 'ground', 'bridge', 'railtrack', 'guardrail')

    OD_CLASSES = ('vehicle', 'pedestrian', 'static', 'dynamic')
    OD_CLASSES_ENC = {'vehicle': 0, 'pedestrian': 1, 'static': 2, 'dynamic': 3}

    SEG_CLASSES = ('unlabeled', 'freespace', 'sidewalk', 'building', 'fence', 'pole', 'terrain',
                   'pedestrian', 'rider', 'vehicle', 'train', 'others', 'roadline')
    SEG_CLASSES_ENC = {'unlabeled': 0, 'freespace': 1, 'sidewalk': 2, 'building': 3, 'fence': 4,
                       'pole': 5, 'terrain': 6, 'pedestrian': 7, 'rider': 8, 'vehicle': 9, 'train': 10,
                       'others': 11, 'roadline': 12}

    ObjectNameMapping = {   # Raw class name -> model class name
        'vehicle': 'vehicle',
        'pedestrian': 'pedestrian',
        'static': 'static',
        'dynamic': 'dynamic',
    }

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='bevdet',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='CAM_FRONT_RGB',
                 stereo=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        self.modality = dict(
            use_camera=True,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        )
        self.ann_file = ann_file
        self.pipeline = pipeline
        self.data_root = data_root
        self.classes = classes
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.box_type_3d = box_type_3d
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.eval_version = eval_version
        self.use_valid_flag = use_valid_flag
        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.stereo = stereo


    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data, key=lambda e: e['timestamp']))
        self.data_infos = data_infos[::self.load_interval]
        return len(self.data_infos)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:     # 需要再读取历史帧的信息
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))

        for k, v in info['cams'].items():
            if v['data_path'].startswith('./'):
                data_path = 'data/' + v['data_path'][2:]
            else:
                data_path = v['data_path']
            input_dict['curr']['cams'][k]['data_path'] = data_path
        gt_labels = [self.OD_CLASSES_ENC[gt_name] for gt_name in input_dict['curr']['gt_names'] if gt_name in self.OD_CLASSES_ENC]
        input_dict['ann_infos'] = [input_dict['ann_infos'], gt_labels]
        input_dict['occ2d_gt_path'] = self.data_infos[index]['cams']['CAM_BEV_SEGMENTATION']['data_path']
        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))     # bevdet4d: [1, ]  只利用前一帧.
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            # 如果使用stereo4d, 不仅当前帧需要利用前一帧图像计算stereo depth, 前一帧也需要利用它的前一帧计算stereo depth.
            # 因此, 我们需要额外读取一帧(也就是前一帧的前一帧).
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def format_results(self,):
        pass

        return None

    def evaluate(self,):
        pass

        return None


    def show(self,):
        pass

        return None
