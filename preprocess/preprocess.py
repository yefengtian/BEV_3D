import os
from .transform import TransformationFactory
from mmcv import Config
from mmdet.utils import compat_cfg
import torch
import numpy as np
from pyquaternion import Quaternion

class PrepareParameter():
    def __init__(
            self,
            data_config,
            params
    ):
        self.data_config = data_config
        self.cam_params = params

    def sample_augmentation(self,H, W,flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        resize = float(fW) / float(W)
        if scale is not None:
            resize += scale
        else:
            resize += self.data_config.get('resize_test', 0.0)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False if flip is None else flip
        rotate = 0
        return resize, crop, flip, rotate

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
        
    def img_transform(self, post_rot, post_tran, resize,
                      crop, flip, rotate):
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        w, x, y, z = info['sensor2ego_rotation'][cam_name]      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(info['sensor2ego_translation'][cam_name])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        return sensor2ego

    def get_inputs(self,flip=None, scale=None):
        sensor2egos = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.data_config['cams']

        for cam_name in cam_names:
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(self.cam_params['cam_intrinsic'])
            sensor2ego = self.get_sensor_transforms(self.cam_params, cam_name)
            H, W = self.cam_params['img_size']
            img_augs = self.sample_augmentation(H,W, flip=flip, scale=scale)
            resize, crop, flip, rotate = img_augs
            post_rot2, post_tran2 = \
                self.img_transform(post_rot,
                                   post_tran,
                                   resize=resize,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）
            
        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        sensor2egos = sensor2egos.unsqueeze(0)
        intrins = torch.stack(intrins)  
        intrins = intrins.unsqueeze(0)# (N_views, 3, 3)
        post_rots = torch.stack(post_rots)    
        post_rots = post_rots.unsqueeze(0)# (N_views, 3, 3)
        post_trans = torch.stack(post_trans)    
        post_trans = post_trans.unsqueeze(0)# (N_views, 3)
        params = [sensor2egos, intrins, post_rots, post_trans]
        return params

def add_params(results,input_params):
    results["img_inputs"][0][1] = input_params[0]
    results["img_inputs"][0][3] = input_params[1]
    results["img_inputs"][0][4] = input_params[2]
    results["img_inputs"][0][5] = input_params[3]
    return results

def preprocess(config,all_data,input_params):
    if config is None:
        config = 'model_interface/config/freespace_occ2d_r50_depth.py'
    cfgs = Config.fromfile(config)
    cfgs = compat_cfg(cfgs)
    transform_result = TransformationFactory.create_transformation_chain(cfgs._cfg_dict['test_pipeline'],all_data)
    results = add_params(transform_result,input_params)
    return results