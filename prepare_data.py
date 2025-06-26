#!/usr/bin/env python3
"""
数据准备脚本
帮助用户准备离线图像数据用于训练和推理
"""

import os
import json
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def create_directory_structure(data_root):
    """创建标准的数据目录结构"""
    dirs = [
        'train',
        'val', 
        'test',
        'train/CAM_FRONT_RGB',
        'train/CAM_LEFT_RGB',
        'train/CAM_RIGHT_RGB',
        'train/CAM_REAR_RGB',
        'train/bev_segmentation',
        'train/annotations',
        'val/CAM_FRONT_RGB',
        'val/CAM_LEFT_RGB',
        'val/CAM_RIGHT_RGB',
        'val/CAM_REAR_RGB',
        'val/bev_segmentation',
        'val/annotations',
        'test/CAM_FRONT_RGB',
        'test/CAM_LEFT_RGB',
        'test/CAM_RIGHT_RGB',
        'test/CAM_REAR_RGB',
        'test/bev_segmentation',
        'test/annotations'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(data_root, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def organize_images(source_dir, target_dir, split_ratio=(0.7, 0.2, 0.1)):
    """组织图像文件到训练/验证/测试集"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    # 查找所有图像文件
    for ext in image_extensions:
        image_files.extend(Path(source_dir).rglob(f'*{ext}'))
        image_files.extend(Path(source_dir).rglob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))  # 去重并排序
    print(f"Found {len(image_files)} image files")
    
    # 计算分割点
    train_end = int(len(image_files) * split_ratio[0])
    val_end = train_end + int(len(image_files) * split_ratio[1])
    
    # 分割数据集
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # 复制文件
    for files, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for img_file in tqdm(files, desc=f"Copying {split_name} files"):
            # 假设图像文件按照相机名称组织
            # 这里需要根据实际的文件命名规则调整
            target_path = os.path.join(target_dir, split_name, 'CAM_FRONT_RGB', img_file.name)
            shutil.copy2(img_file, target_path)

def create_sample_annotation():
    """创建示例标注文件"""
    sample_annotation = {
        "timestamp": 1234567890000,
        "cams": {
            "CAM_FRONT_RGB": {
                "data_path": "./CAM_FRONT_RGB/front_001.jpg",
                "sensor2lidar_rotation": [1.0, 0.0, 0.0, 0.0],
                "sensor2lidar_translation": [0.0, 0.0, 0.0],
                "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
            },
            "CAM_LEFT_RGB": {
                "data_path": "./CAM_LEFT_RGB/left_001.jpg",
                "sensor2lidar_rotation": [0.707, 0.0, 0.707, 0.0],
                "sensor2lidar_translation": [0.0, -1.0, 0.0],
                "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
            },
            "CAM_RIGHT_RGB": {
                "data_path": "./CAM_RIGHT_RGB/right_001.jpg",
                "sensor2lidar_rotation": [0.707, 0.0, -0.707, 0.0],
                "sensor2lidar_translation": [0.0, 1.0, 0.0],
                "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
            },
            "CAM_REAR_RGB": {
                "data_path": "./CAM_REAR_RGB/rear_001.jpg",
                "sensor2lidar_rotation": [0.0, 0.0, 1.0, 0.0],
                "sensor2lidar_translation": [-2.0, 0.0, 0.0],
                "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
            },
            "CAM_BEV_SEGMENTATION": {
                "data_path": "./bev_segmentation/bev_001.png"
            }
        },
        "ann_infos": {
            "gt_names": ["vehicle", "pedestrian"],
            "gt_boxes_3d": [
                [1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0],  # [x, y, z, l, w, h, yaw]
                [3.0, 1.0, 0.5, 0.5, 0.5, 1.7, 0.0]
            ]
        }
    }
    return sample_annotation

def main():
    parser = argparse.ArgumentParser(description='Prepare offline image data for training')
    parser.add_argument('--data_root', type=str, default='data/offline_images', 
                       help='Root directory for data')
    parser.add_argument('--source_dir', type=str, default=None,
                       help='Source directory containing images')
    parser.add_argument('--create_structure', action='store_true',
                       help='Create directory structure')
    parser.add_argument('--organize_images', action='store_true',
                       help='Organize images into train/val/test splits')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample annotation file')
    args = parser.parse_args()

    if args.create_structure:
        print("Creating directory structure...")
        create_directory_structure(args.data_root)

    if args.organize_images and args.source_dir:
        print("Organizing images...")
        organize_images(args.source_dir, args.data_root)

    if args.create_sample:
        print("Creating sample annotation file...")
        sample_ann = create_sample_annotation()
        sample_file = os.path.join(args.data_root, 'sample_annotation.json')
        with open(sample_file, 'w') as f:
            json.dump(sample_ann, f, indent=2)
        print(f"Sample annotation saved to: {sample_file}")

    print("\nData preparation completed!")
    print("\nNext steps:")
    print("1. Place your images in the appropriate camera directories")
    print("2. Create annotation files for each image (optional)")
    print("3. Update camera parameters in utils/cam_params.py if needed")
    print("4. Run training: python train_simple.py model_interface/config/train_config.py")

if __name__ == '__main__':
    main() 