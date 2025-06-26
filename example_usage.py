#!/usr/bin/env python3
"""
BEV 3D感知系统使用示例
演示如何使用离线图像数据进行推理和训练
"""

import os
import sys
import argparse
from pathlib import Path

def example_inference():
    """推理示例"""
    print("=== 离线推理示例 ===")
    print("1. 准备图像数据:")
    print("   python prepare_data.py --create_structure")
    print("   # 将图像文件放入 data/offline_images/test/CAM_FRONT_RGB/ 等目录")
    print()
    print("2. 运行推理:")
    print("   python offline_inference.py --data_root data/offline_images/test --vis ./output")
    print()

def example_training():
    """训练示例"""
    print("=== 模型训练示例 ===")
    print("1. 准备训练数据:")
    print("   python prepare_data.py --create_structure")
    print("   # 将训练图像放入 data/offline_images/train/ 目录")
    print("   # 将验证图像放入 data/offline_images/val/ 目录")
    print()
    print("2. 开始训练:")
    print("   python train_simple.py model_interface/config/train_config.py --gpus 1")
    print()

def example_data_preparation():
    """数据准备示例"""
    print("=== 数据准备示例 ===")
    print("1. 创建目录结构:")
    print("   python prepare_data.py --create_structure")
    print()
    print("2. 组织现有图像:")
    print("   python prepare_data.py --organize_images --source_dir /path/to/images")
    print()
    print("3. 创建示例标注:")
    print("   python prepare_data.py --create_sample")
    print()

def check_environment():
    """检查环境"""
    print("=== 环境检查 ===")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要文件
    required_files = [
        "dataset/offline_image_dataset.py",
        "offline_inference.py",
        "train_simple.py",
        "prepare_data.py",
        "model_interface/config/train_config.py"
    ]
    
    print("\n检查必要文件:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (缺失)")
    
    # 检查数据目录
    data_dirs = [
        "data/offline_images/train",
        "data/offline_images/val",
        "data/offline_images/test"
    ]
    
    print("\n检查数据目录:")
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (未创建)")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='BEV 3D感知系统使用示例')
    parser.add_argument('--check', action='store_true', help='检查环境')
    parser.add_argument('--inference', action='store_true', help='显示推理示例')
    parser.add_argument('--training', action='store_true', help='显示训练示例')
    parser.add_argument('--data', action='store_true', help='显示数据准备示例')
    parser.add_argument('--all', action='store_true', help='显示所有示例')
    
    args = parser.parse_args()
    
    if args.all or not any([args.check, args.inference, args.training, args.data]):
        check_environment()
        example_data_preparation()
        example_inference()
        example_training()
    else:
        if args.check:
            check_environment()
        if args.data:
            example_data_preparation()
        if args.inference:
            example_inference()
        if args.training:
            example_training()

if __name__ == '__main__':
    main() 