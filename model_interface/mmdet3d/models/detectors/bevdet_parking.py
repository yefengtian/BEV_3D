from .bevdepth import BEVDepth

from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head

import torch.nn.functional as F

import numpy as np

@DETECTORS.register_module()
class BEVDepthParking(BEVDepth):
    def __init__(self,
                 occ_head=None,
                 kps_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVDepthParking, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)

        train_cfg = kwargs['train_cfg']
        test_cfg = kwargs['test_cfg']
        pts_train_cfg = train_cfg.pts if train_cfg is not None else None
        pts_test_cfg = test_cfg.pts if test_cfg is not None else None
        kps_head.update(train_cfg=pts_train_cfg, test_cfg=pts_test_cfg)

        self.parkinglot_head = build_head(kps_head)
        self.pts_bbox_head = None
        self.upsample = upsample

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

        if self.occ_head.use_mask:
            mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        else:
            mask_camera = None

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        ##############
        # FreespacePalette = {
        #     0: (0, 0, 0),                # unlabeled, black
        #     1: (169, 169, 169),          # freespace, darkgray
        #     2: (0, 255, 255),            # sidewalks, aqua
        #     3: (100, 149, 237),          # building, cornflowerblue
        #     4: (255, 192, 203),          # fence, pink
        #     5: (255, 255, 0),            # pole, yellow
        #     6: (189, 183, 107),          # terrain, darkkhaki
        #     7: (255, 0, 255),            # pedestrian, fuscia
        #     8: (123, 104, 238),          # rider, mediumslateblue
        #     9: (0, 255, 0),              # vehicle, lime
        #     10: (0, 128, 0),             # train, green
        #     11: (160, 82, 45),           # others, sienna
        #     12: (255, 250, 250)          # roadline, snow
        # }
        # voxel_semantics_vis = voxel_semantics.squeeze(0).squeeze(-1).cpu().numpy()
        # sem_vis = np.zeros((voxel_semantics_vis.shape[0], voxel_semantics_vis.shape[1], 3), dtype=np.uint8)
        # for label, color in FreespacePalette.items():
        #     sem_vis[voxel_semantics_vis == label] = color[::-1]
        # import cv2
        # cv2.imwrite('voxel_semantics.png', sem_vis)
        ##############

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

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
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
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

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        parkinglot_list = self.simple_test_pl([occ_bev_feature], img_metas)
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
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
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
        outs = (self.occ_head(occ_bev_feature), self.parkinglot_head(occ_bev_feature))
        return outs