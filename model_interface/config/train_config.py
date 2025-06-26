_base_ = ['./_base_/nus-3d.py',
          './_base_/default_runtime.py']

point_cloud_range = [-10.0, -10.0, -2.0, 10.0, 10.0, 6.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_RGB', 'CAM_LEFT_RGB', 'CAM_RIGHT_RGB', 'CAM_REAR_RGB'
    ],
    'Ncams': 4,
    'input_size': (544, 960), # 1/16 --> (34, 60)
    'src_size': (1080, 1920), # padding -> (1088, 1920) -> (544, 960)

    # Augmentation
    'resize': (0, 0),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00
}

grid_config = {
    'x': [-10, 10, 0.1],
    'y': [-10, 10, 0.1],
    'z': [-1, 5.4, 6.4],
    'depth': [0.1, 15.0, 0.1]
}

voxel_size = [0.025, 0.025, 0.2]
numC_Trans = 128

model = dict(
    type='BEVDepthParking',  # based on BEVDepthOCC
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        accelerate=False,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    occ_head=dict(
        type='BEVOCCHead2D_V2',
        in_dim=256,
        out_dim=256,
        Dz=1,
        use_mask=False,
        num_classes=13,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CustomFocalLoss',
            use_sigmoid=True,
            loss_weight=1.0)),
    kps_head=dict(
        type='Centerness_Head2D',
        task_specific_weight=[1, 1, 1, 1, 1],
        in_channels=256,
        tasks=[
            dict(num_class=3, class_names=['perpendicular', 'parallel', 'other']),
        ],
        common_heads=dict(
            ctr_offset=(2, 2),
            availability=(3, 2),
            kp0=(2, 2), kp1=(2, 2), kp2=(2, 2), kp3=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointParkingspotBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-15, -15, -5, 15, 15, 5.0],
            max_num=50,
            score_threshold=0.3,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=9,
            nms_kernel_size=15),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_slot=dict(type='L1Loss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[800, 800, 1],
            voxel_size=voxel_size[:2],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=50,
            min_radius=2,
            code_weights=[1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5])),
    test_cfg=dict(
        pts=dict()
    ),
)

# Data
dataset_type = 'OfflineImageDataset'
data_root = 'data/offline_images/'

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0
)

train_pipeline = [
    dict(
        type='PrepareImageInputsV2',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadParkingSpaceFromFile'),
    dict(
        type='LoadDepthCameraFromFile',
        data_config=data_config,
        grid_config=grid_config,
        downsample=1),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'parkinglot_cat', 'parkinglot_sts', 'parkinglot_geom'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputsV2',
        is_train=False,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='LoadOccGTFromFile', is_train=False),
    dict(type='LoadParkingSpaceFromFile', is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=data_config['input_size'][::-1],
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'voxel_semantics',
                                         'parkinglot_cat', 'parkinglot_sts', 'parkinglot_geom'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root + 'train/',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root + 'val/',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root + 'test/',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        use_valid_flag=True,
        box_type_3d='LiDAR'))

# Training settings
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 23])

total_epochs = 24

# Runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

work_dir = './work_dirs/bev_depth_parking'
load_from = None
resume_from = None
workflow = [('train', 1)]

# Evaluation settings
evaluation = dict(interval=1, pipeline=test_pipeline) 