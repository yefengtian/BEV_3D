#!/usr/bin/env python3
"""
下载示例数据脚本
获取适配BEV 3D感知代码格式的示例数据
"""

import os
import requests
import zipfile
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
from PIL import Image

def download_file(url, filename):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def create_sample_images(output_dir, num_samples=10):
    """创建示例图像数据"""
    # 创建相机目录
    cam_dirs = ['CAM_FRONT_RGB', 'CAM_LEFT_RGB', 'CAM_RIGHT_RGB', 'CAM_REAR_RGB']
    for cam_dir in cam_dirs:
        os.makedirs(os.path.join(output_dir, cam_dir), exist_ok=True)
    
    # 创建BEV分割目录
    bev_dir = os.path.join(output_dir, 'bev_segmentation')
    os.makedirs(bev_dir, exist_ok=True)
    
    # 创建标注目录
    ann_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    print(f"Creating {num_samples} sample images...")
    
    for i in range(num_samples):
        # 创建示例图像（模拟不同视角的相机图像）
        img_size = (1920, 1080)
        
        # 前视相机 - 模拟道路场景
        front_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        front_img[:] = (100, 100, 100)  # 灰色背景
        # 添加道路线
        cv2.line(front_img, (0, img_size[1]//2), (img_size[0], img_size[1]//2), (255, 255, 255), 10)
        cv2.line(front_img, (img_size[0]//4, 0), (img_size[0]//4, img_size[1]), (255, 255, 255), 5)
        cv2.line(front_img, (3*img_size[0]//4, 0), (3*img_size[0]//4, img_size[1]), (255, 255, 255), 5)
        
        # 左视相机 - 模拟左侧场景
        left_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        left_img[:] = (80, 120, 80)  # 绿色背景
        cv2.rectangle(left_img, (img_size[0]//3, img_size[1]//3), (2*img_size[0]//3, 2*img_size[1]//3), (255, 0, 0), 3)
        
        # 右视相机 - 模拟右侧场景
        right_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        right_img[:] = (80, 80, 120)  # 蓝色背景
        cv2.circle(right_img, (img_size[0]//2, img_size[1]//2), 100, (0, 255, 0), 5)
        
        # 后视相机 - 模拟后方场景
        rear_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        rear_img[:] = (120, 80, 80)  # 红色背景
        cv2.rectangle(rear_img, (img_size[0]//4, img_size[1]//4), (3*img_size[0]//4, 3*img_size[1]//4), (255, 255, 0), 3)
        
        # 保存图像
        cv2.imwrite(os.path.join(output_dir, 'CAM_FRONT_RGB', f'front_{i:03d}.jpg'), front_img)
        cv2.imwrite(os.path.join(output_dir, 'CAM_LEFT_RGB', f'left_{i:03d}.jpg'), left_img)
        cv2.imwrite(os.path.join(output_dir, 'CAM_RIGHT_RGB', f'right_{i:03d}.jpg'), right_img)
        cv2.imwrite(os.path.join(output_dir, 'CAM_REAR_RGB', f'rear_{i:03d}.jpg'), rear_img)
        
        # 创建BEV分割图像
        bev_size = (400, 400)
        bev_img = np.zeros((bev_size[1], bev_size[0], 3), dtype=np.uint8)
        bev_img[:] = (50, 50, 50)  # 深灰色背景
        
        # 添加道路区域
        cv2.rectangle(bev_img, (50, 150), (350, 250), (128, 128, 128), -1)
        # 添加车辆
        cv2.rectangle(bev_img, (180, 200), (220, 220), (255, 0, 0), -1)
        # 添加行人
        cv2.circle(bev_img, (300, 180), 10, (0, 255, 0), -1)
        
        cv2.imwrite(os.path.join(bev_dir, f'bev_{i:03d}.png'), bev_img)
        
        # 创建标注文件
        annotation = {
            "timestamp": 1234567890000 + i * 100000,
            "cams": {
                "CAM_FRONT_RGB": {
                    "data_path": f"./CAM_FRONT_RGB/front_{i:03d}.jpg",
                    "sensor2lidar_rotation": [1.0, 0.0, 0.0, 0.0],
                    "sensor2lidar_translation": [0.0, 0.0, 0.0],
                    "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
                },
                "CAM_LEFT_RGB": {
                    "data_path": f"./CAM_LEFT_RGB/left_{i:03d}.jpg",
                    "sensor2lidar_rotation": [0.707, 0.0, 0.707, 0.0],
                    "sensor2lidar_translation": [0.0, -1.0, 0.0],
                    "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
                },
                "CAM_RIGHT_RGB": {
                    "data_path": f"./CAM_RIGHT_RGB/right_{i:03d}.jpg",
                    "sensor2lidar_rotation": [0.707, 0.0, -0.707, 0.0],
                    "sensor2lidar_translation": [0.0, 1.0, 0.0],
                    "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
                },
                "CAM_REAR_RGB": {
                    "data_path": f"./CAM_REAR_RGB/rear_{i:03d}.jpg",
                    "sensor2lidar_rotation": [0.0, 0.0, 1.0, 0.0],
                    "sensor2lidar_translation": [-2.0, 0.0, 0.0],
                    "cam_intrinsic": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
                },
                "CAM_BEV_SEGMENTATION": {
                    "data_path": f"./bev_segmentation/bev_{i:03d}.png"
                }
            },
            "ann_infos": {
                "gt_names": ["vehicle", "pedestrian"],
                "gt_boxes_3d": [
                    [0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0],  # 车辆 [x, y, z, l, w, h, yaw]
                    [2.0, -1.0, 0.5, 0.5, 0.5, 1.7, 0.0]   # 行人
                ]
            }
        }
        
        with open(os.path.join(ann_dir, f'ann_{i:03d}.json'), 'w') as f:
            json.dump(annotation, f, indent=2)

def download_kitti_sample():
    """下载KITTI数据集样本"""
    print("正在下载KITTI数据集样本...")
    
    # KITTI数据集的一些公开样本
    kitti_urls = [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
    ]
    
    data_dir = "data/kitti_sample"
    os.makedirs(data_dir, exist_ok=True)
    
    for url in kitti_urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"下载 {filename}...")
            download_file(url, filepath)
            
            # 解压文件
            print(f"解压 {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
    
    print("KITTI样本下载完成！")

def create_nuscenes_sample():
    """创建NuScenes格式的示例数据"""
    print("创建NuScenes格式示例数据...")
    
    data_dir = "data/nuscenes_sample"
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建NuScenes格式的标注文件
    nuscenes_sample = {
        "version": "v1.0-trainval",
        "data_bin": "data_bin",
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "samples": []
    }
    
    # 创建几个样本
    for i in range(5):
        sample = {
            "token": f"sample_{i:03d}",
            "timestamp": 1234567890000 + i * 100000,
            "scene_token": "scene_001",
            "next": f"sample_{(i+1):03d}" if i < 4 else "",
            "prev": f"sample_{(i-1):03d}" if i > 0 else "",
            "data": {
                "CAM_FRONT": f"nuscenes_sample/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512460.jpg",
                "CAM_FRONT_LEFT": f"nuscenes_sample/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603512460.jpg",
                "CAM_FRONT_RIGHT": f"nuscenes_sample/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603512460.jpg",
                "CAM_BACK": f"nuscenes_sample/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603512460.jpg",
                "CAM_BACK_LEFT": f"nuscenes_sample/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603512460.jpg",
                "CAM_BACK_RIGHT": f"nuscenes_sample/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603512460.jpg"
            },
            "anns": []
        }
        nuscenes_sample["samples"].append(sample)
    
    # 保存NuScenes格式文件
    with open(os.path.join(data_dir, "nuscenes_sample.json"), 'w') as f:
        json.dump(nuscenes_sample, f, indent=2)
    
    print("NuScenes格式示例数据创建完成！")

def main():
    parser = argparse.ArgumentParser(description='下载示例数据')
    parser.add_argument('--output_dir', type=str, default='data/sample_data', 
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='样本数量')
    parser.add_argument('--kitti', action='store_true', 
                       help='下载KITTI样本数据')
    parser.add_argument('--nuscenes', action='store_true', 
                       help='创建NuScenes格式样本')
    parser.add_argument('--all', action='store_true', 
                       help='创建所有类型的样本数据')
    
    args = parser.parse_args()
    
    if args.all or not any([args.kitti, args.nuscenes]):
        # 创建基本示例数据
        print("创建基本示例数据...")
        create_sample_images(args.output_dir, args.num_samples)
        
        # 创建训练/验证/测试分割
        for split in ['train', 'val', 'test']:
            split_dir = f"data/offline_images/{split}"
            os.makedirs(split_dir, exist_ok=True)
            
            # 复制部分数据到各个分割
            if split == 'train':
                samples = range(0, int(args.num_samples * 0.7))
            elif split == 'val':
                samples = range(int(args.num_samples * 0.7), int(args.num_samples * 0.9))
            else:  # test
                samples = range(int(args.num_samples * 0.9), args.num_samples)
            
            for i in samples:
                for cam in ['CAM_FRONT_RGB', 'CAM_LEFT_RGB', 'CAM_RIGHT_RGB', 'CAM_REAR_RGB']:
                    src_file = os.path.join(args.output_dir, cam, f'{cam.lower().replace("_rgb", "")}_{i:03d}.jpg')
                    dst_file = os.path.join(split_dir, cam, f'{cam.lower().replace("_rgb", "")}_{i:03d}.jpg')
                    if os.path.exists(src_file):
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        import shutil
                        shutil.copy2(src_file, dst_file)
                
                # 复制BEV分割图像
                src_bev = os.path.join(args.output_dir, 'bev_segmentation', f'bev_{i:03d}.png')
                dst_bev = os.path.join(split_dir, 'bev_segmentation', f'bev_{i:03d}.png')
                if os.path.exists(src_bev):
                    os.makedirs(os.path.dirname(dst_bev), exist_ok=True)
                    import shutil
                    shutil.copy2(src_bev, dst_bev)
                
                # 复制标注文件
                src_ann = os.path.join(args.output_dir, 'annotations', f'ann_{i:03d}.json')
                dst_ann = os.path.join(split_dir, 'annotations', f'ann_{i:03d}.json')
                if os.path.exists(src_ann):
                    os.makedirs(os.path.dirname(dst_ann), exist_ok=True)
                    import shutil
                    shutil.copy2(src_ann, dst_ann)
    
    if args.kitti or args.all:
        download_kitti_sample()
    
    if args.nuscenes or args.all:
        create_nuscenes_sample()
    
    print("\n示例数据创建完成！")
    print(f"数据位置: {args.output_dir}")
    print("\n使用方法:")
    print("1. 运行推理: python offline_inference.py --data_root data/offline_images/test --vis ./output")
    print("2. 开始训练: python train_simple.py model_interface/config/train_config.py --gpus 1")

if __name__ == '__main__':
    main() 