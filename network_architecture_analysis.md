# BEV 3D感知网络架构详细分析

## 1. 网络整体架构概览

这是一个基于BEV（Bird's Eye View）的3D感知网络，主要用于停车场检测和占用栅格预测。网络采用多任务学习架构，同时进行占用栅格分割和停车场关键点检测。

### 1.1 网络类型
- **主模型**: `BEVDepthParking` (基于BEVDepthOCC改进)
- **输入**: 多视角相机图像 (4个相机: 前、左、右、后)
- **输出**: 
  - 占用栅格分割 (13类)
  - 停车场关键点检测 (3类: perpendicular, parallel, other)

### 1.2 数据配置
```python
data_config = {
    'cams': ['CAM_FRONT_RGB', 'CAM_LEFT_RGB', 'CAM_RIGHT_RGB', 'CAM_REAR_RGB'],
    'Ncams': 4,
    'input_size': (544, 960),  # 输入图像尺寸
    'src_size': (1080, 1920),  # 原始图像尺寸
}
```

## 2. 网络组件详细分析

### 2.1 图像编码器 (Image Encoder)

#### 2.1.1 主干网络 (Backbone)
```python
img_backbone=dict(
    type='ResNet',
    depth=50,                    # ResNet-50
    num_stages=4,                # 4个阶段
    out_indices=(2, 3),          # 输出第2、3阶段的特征
    frozen_stages=-1,            # 不冻结任何阶段
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    with_cp=True,                # 使用checkpoint节省内存
    style='pytorch',
    pretrained='torchvision://resnet50',  # 预训练权重
)
```

**功能**: 提取多尺度图像特征
- 输入: 4个相机的图像 [B, 4, 3, H, W]
- 输出: 多尺度特征 [B, 4, C, H', W']

#### 2.1.2 特征金字塔网络 (FPN)
```python
img_neck=dict(
    type='CustomFPN',
    in_channels=[1024, 2048],    # ResNet stage3, stage4的输出通道
    out_channels=256,            # 统一输出通道数
    num_outs=1,                  # 输出1个尺度的特征
    start_level=0,
    out_ids=[0]
)
```

**功能**: 融合多尺度特征，输出统一分辨率的特征图

### 2.2 视图变换器 (View Transformer)

#### 2.2.1 LSS视图变换器
```python
img_view_transformer=dict(
    type='LSSViewTransformerBEVDepth',
    grid_config=grid_config,     # BEV网格配置
    input_size=data_config['input_size'],
    in_channels=256,             # 输入通道数
    out_channels=numC_Trans,     # 输出通道数 (128)
    accelerate=False,            # 是否加速
    loss_depth_weight=1,         # 深度损失权重
    depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
    downsample=16                # 下采样倍数
)
```

**功能**: 将多视角图像特征投影到BEV空间
- 输入: 多视角图像特征 + 相机参数
- 输出: BEV特征 [B, C, H_bev, W_bev]

**关键步骤**:
1. **深度预测**: 为每个像素预测深度值
2. **3D投影**: 将2D图像点投影到3D空间
3. **BEV投影**: 将3D点投影到BEV平面
4. **特征聚合**: 将多视角特征聚合到BEV网格

### 2.3 BEV编码器 (BEV Encoder)

#### 2.3.1 BEV主干网络
```python
img_bev_encoder_backbone=dict(
    type='CustomResNet',
    numC_input=numC_Trans,       # 输入通道数 (128)
    num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]  # [256, 512, 1024]
)
```

**功能**: 在BEV空间提取特征
- 输入: BEV特征 [B, 128, H_bev, W_bev]
- 输出: 多尺度BEV特征

#### 2.3.2 BEV特征融合
```python
img_bev_encoder_neck=dict(
    type='FPN_LSS',
    in_channels=numC_Trans * 8 + numC_Trans * 2,  # 1024 + 256 = 1280
    out_channels=256                              # 统一输出通道
)
```

**功能**: 融合多尺度BEV特征，输出统一分辨率的特征图

### 2.4 任务头 (Task Heads)

#### 2.4.1 占用栅格头 (Occupancy Head)
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

**功能**: 预测BEV空间的占用栅格
- 输入: BEV特征 [B, 256, H_bev, W_bev]
- 输出: 占用栅格 [B, 13, H_bev, W_bev]

#### 2.4.2 停车场关键点头 (Parking Keypoint Head)
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

**功能**: 检测停车场关键点和类型
- 输入: BEV特征 [B, 256, H_bev, W_bev]
- 输出: 
  - 中心点偏移 [B, 2, H_bev, W_bev]
  - 可用性 [B, 3, H_bev, W_bev]
  - 4个关键点 [B, 8, H_bev, W_bev]

## 3. 训练流程详细分析

### 3.1 数据预处理流程

```python
train_pipeline = [
    dict(type='PrepareImageInputsV2', is_train=True, data_config=data_config),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type='LoadOccGTFromFile'),           # 加载占用栅格真值
    dict(type='LoadParkingSpaceFromFile'),    # 加载停车场真值
    dict(type='LoadDepthCameraFromFile', data_config=data_config, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'parkinglot_cat', 'parkinglot_sts', 'parkinglot_geom'])
]
```

### 3.2 损失函数

#### 3.2.1 占用栅格损失
- **类型**: CustomFocalLoss
- **权重**: 1.0
- **作用**: 处理类别不平衡问题

#### 3.2.2 停车场检测损失
- **分类损失**: GaussianFocalLoss (中心点分类)
- **回归损失**: L1Loss (关键点回归)
- **权重**: 0.25

#### 3.2.3 深度损失
- **类型**: 监督深度损失
- **权重**: 1.0

### 3.3 训练配置

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

## 4. 网络架构流程图

```
输入: 4个相机图像 [B, 4, 3, 544, 960]
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   图像编码器 (Image Encoder)                  │
├─────────────────────────────────────────────────────────────┤
│ ResNet-50 Backbone → CustomFPN → 多视角特征 [B, 4, 256, H, W] │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                视图变换器 (View Transformer)                  │
├─────────────────────────────────────────────────────────────┤
│ LSSViewTransformerBEVDepth                                  │
│ ├─ 深度预测网络 (DepthNet)                                   │
│ ├─ 3D投影 (Image → 3D)                                      │
│ ├─ BEV投影 (3D → BEV)                                       │
│ └─ 特征聚合 → BEV特征 [B, 128, 200, 200]                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   BEV编码器 (BEV Encoder)                    │
├─────────────────────────────────────────────────────────────┤
│ CustomResNet → FPN_LSS → BEV特征 [B, 256, 200, 200]         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    任务头 (Task Heads)                       │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────────────────────────┐ │
│ │  占用栅格头     │  │        停车场关键点头               │ │
│ │ BEVOCCHead2D_V2 │  │      Centerness_Head2D             │ │
│ │ 输出: [B,13,H,W]│  │  输出: 中心点+可用性+关键点        │ │
│ └─────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
    ↓
输出: 占用栅格 + 停车场检测结果
```

## 5. 关键技术创新点

### 5.1 LSS视图变换
- **Lift**: 将2D图像特征提升到3D空间
- **Splat**: 将3D特征投影到BEV平面
- **Shoot**: 特征聚合和优化

### 5.2 多任务学习
- 同时进行占用栅格分割和停车场检测
- 共享BEV特征，提高效率

### 5.3 深度监督
- 使用深度真值监督视图变换
- 提高3D投影的准确性

## 6. 训练优化策略

### 6.1 数据增强
- 图像翻转、旋转、缩放
- BEV空间的数据增强

### 6.2 损失平衡
- Focal Loss处理类别不平衡
- 多任务损失权重平衡

### 6.3 内存优化
- Checkpoint机制
- 梯度累积
- 混合精度训练

## 7. 推理流程

1. **图像预处理**: 4个相机图像 → 标准化 → 数据增强
2. **特征提取**: ResNet-50 + FPN → 多尺度特征
3. **视图变换**: LSS → BEV特征
4. **BEV编码**: CustomResNet + FPN → 统一特征
5. **任务推理**: 
   - 占用栅格头 → 13类分割结果
   - 停车场头 → 关键点检测结果
6. **后处理**: NMS → 最终检测结果

这个网络架构充分利用了多视角信息，通过视图变换将2D图像特征转换为BEV表示，实现了高效的3D感知和停车场检测。 