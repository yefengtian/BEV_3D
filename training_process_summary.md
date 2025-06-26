# BEV 3D感知网络训练流程详细总结

## 1. 网络架构概览

### 1.1 核心架构
```
输入: 4个相机图像 [B, 4, 3, 544, 960]
    ↓
图像编码器: ResNet-50 + CustomFPN
    ↓
视图变换器: LSSViewTransformerBEVDepth (2D→3D→BEV)
    ↓
BEV编码器: CustomResNet + FPN_LSS
    ↓
任务头: 占用栅格头 + 停车场关键点头
    ↓
输出: 占用栅格分割 + 停车场检测结果
```

### 1.2 关键参数
- **输入尺寸**: 544×960 (4个相机)
- **BEV网格**: 200×200 (0.025m分辨率)
- **检测范围**: [-10m, -10m, -2m] → [10m, 10m, 6m]
- **占用栅格类别**: 13类
- **停车场类型**: 3类 (perpendicular, parallel, other)

## 2. 训练数据流程

### 2.1 数据预处理管道
```python
train_pipeline = [
    # 1. 图像输入准备
    dict(type='PrepareImageInputsV2', 
         is_train=True, 
         data_config=data_config),
    
    # 2. 加载BEV深度标注
    dict(type='LoadAnnotationsBEVDepth', 
         bda_aug_conf=bda_aug_conf, 
         classes=class_names),
    
    # 3. 加载占用栅格真值
    dict(type='LoadOccGTFromFile'),
    
    # 4. 加载停车场真值
    dict(type='LoadParkingSpaceFromFile'),
    
    # 5. 加载深度相机数据
    dict(type='LoadDepthCameraFromFile', 
         data_config=data_config, 
         grid_config=grid_config),
    
    # 6. 格式化数据
    dict(type='DefaultFormatBundle3D', 
         class_names=class_names),
    
    # 7. 收集数据
    dict(type='Collect3D', 
         keys=['img_inputs', 'gt_depth', 'voxel_semantics', 
               'parkinglot_cat', 'parkinglot_sts', 'parkinglot_geom'])
]
```

### 2.2 数据增强策略
```python
bda_aug_conf = dict(
    rot_lim=(-0., 0.),      # 旋转限制
    scale_lim=(1., 1.),     # 缩放限制
    flip_dx_ratio=0.0,      # X轴翻转概率
    flip_dy_ratio=0.0       # Y轴翻转概率
)
```

## 3. 网络组件详解

### 3.1 图像编码器 (Image Encoder)

#### ResNet-50 Backbone
```python
img_backbone=dict(
    type='ResNet',
    depth=50,                    # ResNet-50
    num_stages=4,                # 4个阶段
    out_indices=(2, 3),          # 输出stage2, stage3特征
    frozen_stages=-1,            # 不冻结任何阶段
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    with_cp=True,                # 使用checkpoint节省内存
    style='pytorch',
    pretrained='torchvision://resnet50',
)
```

**功能**: 提取多尺度图像特征
- 输入: [B, 4, 3, 544, 960]
- 输出: 多尺度特征 [B, 4, C, H', W']

#### CustomFPN
```python
img_neck=dict(
    type='CustomFPN',
    in_channels=[1024, 2048],    # ResNet stage3, stage4输出
    out_channels=256,            # 统一输出通道
    num_outs=1,                  # 输出1个尺度
    start_level=0,
    out_ids=[0]
)
```

**功能**: 融合多尺度特征，输出统一分辨率

### 3.2 视图变换器 (View Transformer)

#### LSSViewTransformerBEVDepth
```python
img_view_transformer=dict(
    type='LSSViewTransformerBEVDepth',
    grid_config=grid_config,     # BEV网格配置
    input_size=data_config['input_size'],
    in_channels=256,             # 输入通道
    out_channels=numC_Trans,     # 输出通道 (128)
    accelerate=False,            # 是否加速
    loss_depth_weight=1,         # 深度损失权重
    depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
    downsample=16                # 下采样倍数
)
```

**核心步骤**:
1. **深度预测**: DepthNet预测每个像素的深度
2. **3D投影**: 2D图像点 → 3D空间点
3. **BEV投影**: 3D点 → BEV平面
4. **特征聚合**: 多视角特征聚合到BEV网格

### 3.3 BEV编码器 (BEV Encoder)

#### CustomResNet
```python
img_bev_encoder_backbone=dict(
    type='CustomResNet',
    numC_input=numC_Trans,       # 输入通道 (128)
    num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]  # [256, 512, 1024]
)
```

#### FPN_LSS
```python
img_bev_encoder_neck=dict(
    type='FPN_LSS',
    in_channels=numC_Trans * 8 + numC_Trans * 2,  # 1024 + 256 = 1280
    out_channels=256                              # 统一输出通道
)
```

**功能**: 在BEV空间提取和融合特征

### 3.4 任务头 (Task Heads)

#### 占用栅格头 (Occupancy Head)
```python
occ_head=dict(
    type='BEVOCCHead2D_V2',
    in_dim=256,                  # 输入维度
    out_dim=256,                 # 输出维度
    Dz=1,                        # Z轴维度 (2D占用栅格)
    use_mask=False,
    num_classes=13,              # 13个类别
    use_predicter=True,
    class_balance=True,
    loss_occ=dict(
        type='CustomFocalLoss',  # Focal Loss
        use_sigmoid=True,
        loss_weight=1.0
    )
)
```

**输出**: [B, 13, H_bev, W_bev] - 13类占用栅格分割

#### 停车场关键点头 (Parking Keypoint Head)
```python
kps_head=dict(
    type='Centerness_Head2D',
    task_specific_weight=[1, 1, 1, 1, 1],
    in_channels=256,
    tasks=[
        dict(num_class=3, class_names=['perpendicular', 'parallel', 'other']),
    ],
    common_heads=dict(
        ctr_offset=(2, 2),       # 中心点偏移
        availability=(3, 2),     # 可用性 (vacant, vehicle-occupied, other-occupied)
        kp0=(2, 2), kp1=(2, 2), kp2=(2, 2), kp3=(2, 2)  # 4个关键点
    ),
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
        nms_kernel_size=15
    ),
    separate_head=dict(
        type='SeparateHead', init_bias=-2.19, final_kernel=3
    ),
    loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
    loss_slot=dict(type='L1Loss', reduction='mean', loss_weight=0.25)
)
```

**输出**: 
- 中心点偏移: [B, 2, H_bev, W_bev]
- 可用性: [B, 3, H_bev, W_bev]  
- 4个关键点: [B, 8, H_bev, W_bev]

## 4. 损失函数设计

### 4.1 总损失函数
```
Total Loss = L_occupancy + L_parking + L_depth
```

### 4.2 各损失函数详解

#### 占用栅格损失 (Occupancy Loss)
- **类型**: CustomFocalLoss
- **权重**: 1.0
- **作用**: 处理13类占用栅格的类别不平衡问题

#### 停车场检测损失 (Parking Loss)
- **分类损失**: GaussianFocalLoss (中心点分类)
- **回归损失**: L1Loss (关键点回归)
- **权重**: 0.25
- **作用**: 检测停车场类型和关键点位置

#### 深度损失 (Depth Loss)
- **类型**: 监督深度损失
- **权重**: 1.0
- **作用**: 监督视图变换中的深度预测

## 5. 训练配置

### 5.1 训练参数
```python
train_cfg=dict(
    pts=dict(
        point_cloud_range=point_cloud_range,  # [-10, -10, -2, 10, 10, 6]
        grid_size=[800, 800, 1],              # BEV网格大小
        voxel_size=voxel_size[:2],            # [0.025, 0.025]
        out_size_factor=4,                    # 输出尺寸因子
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=50,                          # 最大目标数
        min_radius=2,
        code_weights=[1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5]
    )
)
```

### 5.2 数据配置
```python
data = dict(
    samples_per_gpu=6,          # 每GPU样本数
    workers_per_gpu=6,          # 每GPU工作进程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'train_30.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
    )
)
```

## 6. 训练优化策略

### 6.1 内存优化
- **Checkpoint机制**: `with_cp=True` 节省显存
- **梯度累积**: 支持大batch size训练
- **混合精度训练**: 使用FP16加速训练

### 6.2 数据增强
- **图像增强**: 翻转、旋转、缩放
- **BEV增强**: 在BEV空间进行数据增强
- **深度增强**: 深度数据的随机扰动

### 6.3 损失平衡
- **Focal Loss**: 处理类别不平衡
- **权重调整**: 多任务损失权重平衡
- **动态权重**: 根据训练进度调整权重

## 7. 推理流程

### 7.1 前向推理步骤
1. **图像预处理**: 4个相机图像 → 标准化
2. **特征提取**: ResNet-50 + FPN → 多尺度特征
3. **视图变换**: LSS → BEV特征 [B, 128, 200, 200]
4. **BEV编码**: CustomResNet + FPN → 统一特征 [B, 256, 200, 200]
5. **任务推理**: 
   - 占用栅格头 → 13类分割结果
   - 停车场头 → 关键点检测结果
6. **后处理**: NMS → 最终检测结果

### 7.2 后处理
- **NMS**: 非极大值抑制去除重复检测
- **阈值过滤**: 根据置信度过滤低质量检测
- **结果融合**: 占用栅格和停车场检测结果融合

## 8. 关键技术特点

### 8.1 LSS视图变换
- **Lift**: 2D图像特征 → 3D空间特征
- **Splat**: 3D特征 → BEV平面特征
- **Shoot**: 特征聚合和优化

### 8.2 多任务学习
- 同时进行占用栅格分割和停车场检测
- 共享BEV特征，提高计算效率
- 平衡多任务损失权重

### 8.3 深度监督
- 使用深度真值监督视图变换
- 提高3D投影的准确性
- 增强网络的空间感知能力

## 9. 性能指标

### 9.1 占用栅格评估
- **IoU**: 各类别的交并比
- **mIoU**: 平均交并比
- **准确率**: 像素级分类准确率

### 9.2 停车场检测评估
- **mAP**: 平均精度
- **召回率**: 检测召回率
- **精确率**: 检测精确率

### 9.3 整体性能
- **推理速度**: FPS (帧率)
- **内存占用**: GPU显存使用量
- **模型大小**: 参数量

这个网络架构充分利用了多视角信息，通过视图变换将2D图像特征转换为BEV表示，实现了高效的3D感知和停车场检测，是一个典型的多任务学习网络。 