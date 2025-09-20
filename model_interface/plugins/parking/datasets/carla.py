import tempfile
from os import path as osp
from typing import List, Dict, Any

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

# 使用官方 mmdet3d 的 registry
from mmdet3d.registry import DATASETS

# --- 兼容导入：不同版本 mmdet3d 的 Det3DDataset 路径可能不同 ---
try:
    # 常规入口（>=1.1 常见）
    from mmdet3d.datasets import Det3DDataset
except Exception:
    # 兜底到文件路径（某些版本）
    from mmdet3d.datasets.det3d_dataset import Det3DDataset
# ------------------------------------------------------------

import os.path as osp
import pickle
from typing import List, Dict, Any


@DATASETS.register_module()
class CarlaDataset(Det3DDataset):
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
        ann = self.ann_file if osp.isabs(self.ann_file) else osp.join(self.data_root, self.ann_file)
        with open(ann, 'rb') as f:
            raw = pickle.load(f)

        # 这里根据你的 pkl 结构做适配：
        if isinstance(raw, list):
            data_list = raw
        elif isinstance(raw, dict) and 'data_list' in raw:
            data_list = raw['data_list']
        else:
            raise ValueError(f'Unknown annotation format: {ann}')

        # 如果你的 pipeline 需要特定键名（比如 'img_inputs'、'voxel_semantics' 等），
        # 可以在这里做一次规范化/补充。
        return data_list
    
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
