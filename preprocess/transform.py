from mmcv.parallel import DataContainer as DC
import numpy as np
import torch
import os
from PIL import Image
from pyquaternion import Quaternion

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

class PrepareImageInputsV2(object):
    def __init__(
            self,
            data_config,
    ):
        self.data_config = data_config
        self.normalize_img = mmlabNormalize

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
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
        return resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image

        # HACK for carla dataset
        if img.size == (1920, 1088):
            img = np.array(img)
            img = img[::2, ::2, ...]
            img = Image.fromarray(img)
        else:
            img = img.resize(resize_dims)

        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img,resize_dims, crop, flip, rotate):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        return img

    def get_ego_transforms(self, info, cam_name):
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return ego2global
    
    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # HACK for carla dataset
            if 'carla' in filename and img.size == (1920, 1080):
                # padding from 1920x1080 to 1920x1088 by adding black pixels at both top and bottom
                img_new = Image.new('RGB', (1920, 1088), (0, 0, 0))
                img_new.paste(img, (0, 4))
                img = img_new
            # image view augmentation (resize, crop, horizontal flip, rotate)
            ego2global = self.get_ego_transforms(results['curr'], cam_name)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize_dims, crop, flip, rotate = img_augs
            img= self.img_transform(img,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)
            imgs.append(self.normalize_img(img))
            ego2globals.append(ego2global) 
            
        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
        imgs = imgs.unsqueeze(0)
        sensor2egos = None     # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)       # (N_views, 4, 4)
        ego2globals = ego2globals.unsqueeze(0)
        intrins = None            # (N_views, 3, 3)
        post_rots = None       # (N_views, 3, 3)
        post_trans = None     # (N_views, 3)

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results

class LoadAnnotationsBEVDepth():
    def __init__(self, bda_aug_conf):
        self.bda_aug_conf = bda_aug_conf

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""

        rotate_bda = 0
        scale_bda = 1.0
        flip_dx = False
        flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        rot_mat = rot_mat.unsqueeze(0)
        return rot_mat

    def __call__(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_rot = self.bev_transform(rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                 post_trans, bda_rot)
        return results
    
class Collect3D(object):
    def __init__(
        self,
        keys,
        meta_keys=('flip','pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d','box_type_3d','pcd_scale_factor')):
        
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            img_metas[key] = None
            
        # repeated_img_metas = [img_metas[key] for _ in range(6)]
        data['img_metas'] = [DC([[img_metas]], cpu_only=True)]
        # results['gt_depth'] = None
        results['voxel_semantics'] = None
        results['parkinglot_cat'] = None
        results['parkinglot_sts'] = None
        results['parkinglot_geom'] = None
        for key in self.keys:
            data[key] = results[key]
        data['img_inputs'] = [list(data['img_inputs'])]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'
            
class TransformationFactory:
    @staticmethod
    def create_transformation_chain(configs,data):
        results = None

        for config in configs:
            if config['type'] == 'PrepareImageInputsV2':
                transform = PrepareImageInputsV2(data_config = config['data_config'])
                results = transform(data)
            elif config['type'] == 'LoadAnnotationsBEVDepth':
                transform = LoadAnnotationsBEVDepth(config['bda_aug_conf'])
                results = transform(results)
            elif config['type'] == 'MultiScaleFlipAug3D':
                keys=['img_inputs', 'voxel_semantics', 'parkinglot_cat','parkinglot_sts', 'parkinglot_geom']
                transform = Collect3D(keys=keys)
                results = transform(results)
            else:
                pass
            
        return results



