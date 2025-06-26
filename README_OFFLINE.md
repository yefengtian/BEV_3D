# BEV 3D感知 - 离线图像版本

本项目是基于MMDet3D的BEV（Bird's Eye View）3D感知系统，支持离线图像数据的训练和推理。

## 主要修改

### 1. 输入数据修改
- **原版本**: 使用Carla实时数据，通过Redis传输
- **新版本**: 支持离线图像文件，直接从本地目录读取

### 2. 新增功能
- 离线图像数据集类 (`dataset/offline_image_dataset.py`)
- 离线推理脚本 (`offline_inference.py`)
- 训练脚本 (`train_simple.py`)
- 数据准备工具 (`prepare_data.py`)
- 训练配置文件 (`model_interface/config/train_config.py`)

## 快速开始

### 1. 准备数据

```bash
# 创建数据目录结构
python prepare_data.py --create_structure

# 组织图像文件（如果有源目录）
python prepare_data.py --organize_images --source_dir /path/to/your/images

# 创建示例标注文件
python prepare_data.py --create_sample
```

### 2. 数据目录结构

```
data/offline_images/
├── train/
│   ├── CAM_FRONT_RGB/
│   ├── CAM_LEFT_RGB/
│   ├── CAM_RIGHT_RGB/
│   ├── CAM_REAR_RGB/
│   ├── bev_segmentation/
│   └── annotations/
├── val/
│   ├── CAM_FRONT_RGB/
│   ├── CAM_LEFT_RGB/
│   ├── CAM_RIGHT_RGB/
│   ├── CAM_REAR_RGB/
│   ├── bev_segmentation/
│   └── annotations/
└── test/
    ├── CAM_FRONT_RGB/
    ├── CAM_LEFT_RGB/
    ├── CAM_RIGHT_RGB/
    ├── CAM_REAR_RGB/
    ├── bev_segmentation/
    └── annotations/
```

### 3. 离线推理

```bash
# 基本推理
python offline_inference.py --data_root /path/to/your/images --vis ./output

# 指定模型和配置
python offline_inference.py \
    --config model_interface/config/freespace_occ2d_r50_depth.py \
    --weights model_interface/ckpts/epoch_69.pth \
    --data_root /path/to/your/images \
    --vis ./output \
    --start_idx 0 \
    --end_idx 100
```

### 4. 训练模型

```bash
# 开始训练
python train_simple.py model_interface/config/train_config.py --gpus 1

# 指定工作目录
python train_simple.py model_interface/config/train_config.py \
    --work-dir ./work_dirs/my_training \
    --gpus 2

# 恢复训练
python train_simple.py model_interface/config/train_config.py \
    --resume-from ./work_dirs/my_training/latest.pth
```

## 配置说明

### 相机参数
在 `utils/cam_params.py` 中配置相机参数：
- 相机内参矩阵
- 相机外参（相对于车辆坐标系）
- 图像尺寸

### 训练配置
在 `model_interface/config/train_config.py` 中配置：
- 数据路径
- 模型参数
- 训练超参数
- 数据增强策略

## 数据格式

### 图像文件
- 支持格式：JPG, JPEG, PNG, BMP
- 建议尺寸：1920x1080 或 960x544
- 命名规则：按时间戳或序列号命名

### 标注文件（可选）
```json
{
    "timestamp": 1234567890000,
    "cams": {
        "CAM_FRONT_RGB": {
            "data_path": "./CAM_FRONT_RGB/front_001.jpg",
            "sensor2lidar_rotation": [1.0, 0.0, 0.0, 0.0],
            "sensor2lidar_translation": [0.0, 0.0, 0.0],
            "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
        }
    },
    "ann_infos": {
        "gt_names": ["vehicle", "pedestrian"],
        "gt_boxes_3d": [
            [1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0]
        ]
    }
}
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- MMDet3D
- MMCV
- 其他依赖见 `bevdet_pip_env.txt`

## 注意事项

1. **相机标定**: 确保相机内参和外参准确，这对BEV投影至关重要
2. **数据质量**: 图像质量直接影响模型性能，建议使用高质量图像
3. **标注质量**: 如果有标注数据，确保标注准确性和一致性
4. **硬件要求**: 训练需要GPU，推理可以在CPU上运行但速度较慢

## 故障排除

### 常见问题

1. **内存不足**: 减少batch size或图像尺寸
2. **CUDA错误**: 检查GPU驱动和PyTorch版本兼容性
3. **数据加载错误**: 检查文件路径和格式是否正确
4. **模型加载失败**: 确认权重文件路径和模型配置匹配

### 调试技巧

- 使用小数据集进行测试
- 检查日志输出
- 验证数据预处理步骤
- 确认配置文件参数

## 扩展功能

- 支持更多相机配置
- 添加数据增强策略
- 支持多GPU训练
- 添加模型评估工具
- 支持导出ONNX模型

## 联系方式

如有问题，请查看项目文档或提交Issue。 