import tempfile
from os import path as osp
from typing import List, Dict, Any

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

# 使用官方 mmdet3d 的 registry
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Base3DDataset


@DATASETS.register_module()
class CarlaDataset(Base3DDataset):
    """Carla 数据集类 - 简化版本"""
    
    METAINFO = dict(CLASSES=('vehicle', 'pedestrian', 'static', 'dynamic'))
    
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
                 stereo=False,
                 **kwargs):
        
        # 设置基本属性
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.stereo = stereo
        
        # 调用父类初始化
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        
        print("CarlaDataset 初始化完成")

    def load_data_list(self) -> List[Dict[str, Any]]:
        """加载数据列表"""
        # 使用父类的方法加载数据
        return super().load_data_list()
    
    def parse_data_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """解析数据信息"""
        # 简化的数据解析，返回基本的信息
        data_info = {
            'sample_idx': info.get('sample_idx', ''),
            'timestamp': info.get('timestamp', 0.0),
        }
        
        # 如果有相机信息
        if 'cams' in info:
            data_info['curr'] = info
            # 处理相机路径
            for k, v in info['cams'].items():
                if v['data_path'].startswith('./'):
                    data_info['curr']['cams'][k]['data_path'] = 'data/' + v['data_path'][2:]
        
        return data_info
