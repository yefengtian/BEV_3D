import os
import mmcv
import numpy as np
import pyquaternion
from pathlib import Path
import json

class OfflineImageDataset():
    """离线图像数据集类，用于处理本地图像文件"""
    
    OD_CLASSES = ('vehicle', 'pedestrian', 'static', 'dynamic')
    OD_CLASSES_ENC = {'vehicle': 0, 'pedestrian': 1, 'static': 2, 'dynamic': 3}

    SEG_CLASSES = ('unlabeled', 'freespace', 'sidewalk', 'building', 'fence', 'pole', 'terrain',
                   'pedestrian', 'rider', 'vehicle', 'train', 'others', 'roadline')
    SEG_CLASSES_ENC = {'unlabeled': 0, 'freespace': 1, 'sidewalk': 2, 'building': 3, 'fence': 4,
                       'pole': 5, 'terrain': 6, 'pedestrian': 7, 'rider': 8, 'vehicle': 9, 'train': 10,
                       'others': 11, 'roadline': 12}

    def __init__(self,
                 data_root,
                 annotation_file=None,
                 pipeline=None,
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
        
        self.data_root = data_root
        self.annotation_file = annotation_file
        self.pipeline = pipeline
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

        self.modality = dict(
            use_camera=True,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        )

        # 相机配置
        self.cam_names = ['CAM_FRONT_RGB', 'CAM_LEFT_RGB', 'CAM_RIGHT_RGB', 'CAM_REAR_RGB']
        
        # 加载数据
        self.data_infos = self.load_annotations()
        
    def load_annotations(self):
        """加载数据标注信息"""
        if self.annotation_file and os.path.exists(self.annotation_file):
            data = mmcv.load(self.annotation_file, file_format='pkl')
            return list(sorted(data, key=lambda e: e['timestamp']))
        else:
            return self._generate_from_directory()
    
    def _generate_from_directory(self):
        """从目录结构自动生成数据信息"""
        data_infos = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(self.data_root).rglob(f'*{ext}'))
        
        image_files.sort()
        
        for i, img_file in enumerate(image_files):
            timestamp = i * 1000000
            info = {
                'timestamp': timestamp,
                'cams': {}
            }
            
            for cam_name in self.cam_names:
                cam_dir = img_file.parent / cam_name.lower()
                if cam_dir.exists():
                    cam_img_file = cam_dir / img_file.name
                    if cam_img_file.exists():
                        info['cams'][cam_name] = {
                            'data_path': str(cam_img_file.relative_to(self.data_root)),
                            'sensor2lidar_rotation': [1.0, 0.0, 0.0, 0.0],
                            'sensor2lidar_translation': [0.0, 0.0, 0.0],
                            'cam_intrinsic': [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
                        }
            
            data_infos.append(info)
        
        return data_infos

    def get_cat_ids(self, idx):
        """获取类别分布"""
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info.get('gt_names', []))

        cat_ids = []
        for name in gt_names:
            if name in self.OD_CLASSES:
                cat_ids.append(self.OD_CLASSES_ENC[name])
        return cat_ids

    def get_data_info(self, index):
        """获取数据信息"""
        info = self.data_infos[index]
        input_dict = dict(timestamp=info['timestamp'] / 1e6)
        
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        
        input_dict.update(dict(curr=info))
        
        for k, v in info['cams'].items():
            if v['data_path'].startswith('./'):
                data_path = 'data/' + v['data_path'][2:]
            else:
                data_path = v['data_path']
            input_dict['curr']['cams'][k]['data_path'] = data_path
        
        return input_dict

    def get_ann_info(self, index):
        """获取标注信息"""
        info = self.data_infos[index]
        
        if 'ann_infos' not in info:
            return None
            
        annos = info['ann_infos']
        gt_names = annos.get('gt_names', [])
        gt_boxes_3d = annos.get('gt_boxes_3d', [])
        
        return {
            'gt_names': gt_names,
            'gt_boxes_3d': gt_boxes_3d,
        }

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.get_data_info(idx) 