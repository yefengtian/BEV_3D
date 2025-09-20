import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

# 使用官方 mmdet3d 的 registry
from mmdet3d.registry import MODELS
from mmdet3d.models.builder import build_head

import numpy as np


@MODELS.register_module()
class BEVDepthParking:
    """BEVDepthParking 模型，基于 BEVDepth 实现停车位检测"""
    
    def __init__(self,
                 img_backbone,
                 img_neck, 
                 img_view_transformer,
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 occ_head=None,
                 kps_head=None,
                 upsample=False,
                 pts_bbox_head=None,
                 **kwargs):
        
        # 初始化基础组件
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.img_view_transformer = MODELS.build(img_view_transformer)
        self.img_bev_encoder_backbone = MODELS.build(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)
        
        # 初始化头部
        if occ_head:
            self.occ_head = MODELS.build(occ_head)
        else:
            self.occ_head = None
            
        if kps_head:
            train_cfg = kwargs.get('train_cfg')
            test_cfg = kwargs.get('test_cfg')
            pts_train_cfg = train_cfg.pts if train_cfg is not None else None
            pts_test_cfg = test_cfg.pts if test_cfg is not None else None
            kps_head.update(train_cfg=pts_train_cfg, test_cfg=pts_test_cfg)
            self.parkinglot_head = MODELS.build(kps_head)
        else:
            self.parkinglot_head = None
            
        self.pts_bbox_head = MODELS.build(pts_bbox_head) if pts_bbox_head else None
        self.upsample = upsample
        
        # 设置属性
        self.with_img_neck = img_neck is not None
        self.with_pts_bbox = pts_bbox_head is not None

    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(imgs)    # x: (B, N, C, fH, fW)
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)

        x, depth = self.img_view_transformer([x, sensor2keyegos, ego2globals, intrins, post_rots,
                                              post_trans, bda, mlp_input])
        # x: (B, C, Dy, Dx)
        # depth: (B*N, D, fH, fW)
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)

        if self.occ_head and self.occ_head.use_mask:
            mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        else:
            mask_camera = None

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        if self.occ_head:
            loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
            losses.update(loss_occ)

        if self.parkinglot_head:
            pl_cat = kwargs['parkinglot_cat']
            pl_sts = kwargs['parkinglot_sts']
            pl_geom = kwargs['parkinglot_geom']

            loss_parkinglot = self.forward_pl_train([occ_bev_feature], pl_cat, pl_sts, pl_geom)
            losses.update(loss_parkinglot)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def forward_pl_train(self, img_feats, pl_cat, pl_sts, pl_geom):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            parkinglot_gt: dict
        """
        outs = self.parkinglot_head(img_feats)
        loss_parkinglot = self.parkinglot_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            pl_cat, pl_sts, pl_geom
        )
        return loss_parkinglot

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas) if self.occ_head else []
        parkinglot_list = self.simple_test_pl([occ_bev_feature], img_metas) if self.parkinglot_head else []
        return occ_list, parkinglot_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def simple_test_pl(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:
        Returns:
            parkinglot_preds: List[List[dict], List[dict], ...]
        """
        outs = self.parkinglot_head(img_feats)
        parkinglot_preds = self.parkinglot_head.get_bboxes(outs, img_metas)      # List[List[dict], List[dict], ...]
        return parkinglot_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = []
        if self.occ_head:
            outs.append(self.occ_head(occ_bev_feature))
        if self.parkinglot_head:
            outs.append(self.parkinglot_head(occ_bev_feature))
        return tuple(outs) if outs else None
