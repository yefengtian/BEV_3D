# 导入自定义插件
custom_imports = dict(
    imports=[
        'model_interface.plugins.parking',  # 导入停车位检测插件
    ],
    allow_failed_imports=False,
)

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

#------------Distributed config------------------------
dist_params = dict(backend='nccl')
opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = False

model = dict(
    type='BEVDepthParking',  # 使用组合包装器
    inner_model=dict(
        type='BEVDepth',  # 使用官方的 BEVDepth 作为内部模型
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
            type='FPN',
            in_channels=[1024, 2048],
            out_channels=256,
            num_outs=1,
            start_level=0,
            add_extra_convs='on_output'),
        img_view_transformer=dict(
            type='LSSViewTransformer',
            grid_config=grid_config,
            input_size=data_config['input_size'],
            in_channels=256,
            out_channels=numC_Trans,
            downsample=16),
        img_bev_encoder_backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=3,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            style='pytorch'),
        img_bev_encoder_neck=dict(
            type='FPN',
            in_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
            out_channels=256,
            num_outs=1),
        pts_bbox_head=dict(
            type='CenterHead',
            in_channels=256,
            tasks=[
                dict(num_class=1, class_names=['car']),
            ],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=9),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True),
        # model training and testing settings
        train_cfg=dict(
            pts=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[800, 800, 1],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
        test_cfg=dict(
            pts=dict(
                pc_range=point_cloud_range[:2],
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                post_max_size=83,
                max_per_img=500,
                max_pool_nms=False,
                use_rotate_nms=True,
                nms_thr=0.2,
                score_thr=0.0,
                min_bbox_size=0,
                use_scale_nms=True,
                max_num=500))
    ),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
)

# Data
dataset_type = 'CarlaDataset'
data_root = 'data/carla_bev/'

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
    dict(type='LoadOccGTFromFile',
         is_train=False),
    dict(type='LoadParkingSpaceFromFile',
         is_train=False),
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


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'test_30.pkl')

# work_dir = '/home/zbz/ws/BEVParking/work_dirs/freespace_occ2d_r50_depth_1127'

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'train_70.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_by_epoch=True,
    warmup_ratio=0.001,
    min_lr_ratio=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=300)

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
]

# load_from = "ckpts/bevdet-r50-cbgs.pth"
evaluation = dict(interval=1, start=301, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)