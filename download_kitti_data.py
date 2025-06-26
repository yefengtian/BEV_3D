#!/usr/bin/env python3
"""
KITTI数据集下载和处理脚本
下载真实的KITTI数据集并转换为项目所需的格式
"""

import os
import requests
import zipfile
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

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

def download_kitti_dataset():
    """下载KITTI数据集"""
    print("正在下载KITTI数据集...")
    
    # KITTI数据集文件
    kitti_files = [
        {
            "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
            "name": "image_2",
            "desc": "左侧彩色图像"
        },
        {
            "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip", 
            "name": "image_3",
            "desc": "右侧彩色图像"
        },
        {
            "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
            "name": "calib",
            "desc": "相机标定参数"
        },
        {
            "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
            "name": "label_2", 
            "desc": "2D标注"
        }
    ]
    
    data_dir = "data/kitti_raw"
    os.makedirs(data_dir, exist_ok=True)
    
    for file_info in kitti_files:
        url = file_info["url"]
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"下载 {file_info['desc']}...")
            if download_file(url, filepath):
                # 解压文件
                print(f"解压 {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
            else:
                print(f"下载失败: {filename}")
        else:
            print(f"文件已存在: {filename}")
    
    print("KITTI数据集下载完成！")

def load_calib_file(calib_file):
    """加载标定文件"""
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    calib_data = {}
    for line in lines:
        key, value = line.split(':', 1)
        calib_data[key] = np.array([float(x) for x in value.split()])
    
    return calib_data

def load_label_file(label_file):
    """加载标注文件"""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 15:
            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(x) for x in parts[4:8]],  # 2D边界框
                'dimensions': [float(x) for x in parts[8:11]],  # 3D尺寸
                'location': [float(x) for x in parts[11:14]],  # 3D位置
                'rotation_y': float(parts[14])  # 旋转角
            }
            objects.append(obj)
    
    return objects

def convert_kitti_to_project_format(kitti_dir, output_dir, num_samples=50):
    """将KITTI数据转换为项目格式"""
    print(f"转换KITTI数据，处理 {num_samples} 个样本...")
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['CAM_FRONT_RGB', 'CAM_LEFT_RGB', 'CAM_RIGHT_RGB', 'CAM_REAR_RGB', 'bev_segmentation', 'annotations']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # 获取所有图像文件
    image_2_dir = os.path.join(kitti_dir, 'training', 'image_2')
    image_3_dir = os.path.join(kitti_dir, 'training', 'image_3')
    calib_dir = os.path.join(kitti_dir, 'training', 'calib')
    label_dir = os.path.join(kitti_dir, 'training', 'label_2')
    
    if not all(os.path.exists(d) for d in [image_2_dir, image_3_dir, calib_dir, label_dir]):
        print("KITTI数据目录不完整，请先下载完整数据集")
        return
    
    # 获取图像文件列表
    image_files = sorted([f for f in os.listdir(image_2_dir) if f.endswith('.png')])
    num_samples = min(num_samples, len(image_files))
    
    # 分割数据集
    train_end = int(num_samples * 0.7)
    val_end = train_end + int(num_samples * 0.2)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:num_samples]
    }
    
    for split_name, split_files in splits.items():
        print(f"处理 {split_name} 集 ({len(split_files)} 个样本)...")
        
        for i, filename in enumerate(tqdm(split_files, desc=f"处理 {split_name}")):
            sample_id = filename.split('.')[0]
            
            # 加载标定数据
            calib_file = os.path.join(calib_dir, f'{sample_id}.txt')
            if not os.path.exists(calib_file):
                continue
                
            calib_data = load_calib_file(calib_file)
            
            # 加载标注数据
            label_file = os.path.join(label_dir, f'{sample_id}.txt')
            objects = []
            if os.path.exists(label_file):
                objects = load_label_file(label_file)
            
            # 复制图像文件
            # 前视相机（左相机）
            src_img_2 = os.path.join(image_2_dir, filename)
            dst_img_2 = os.path.join(output_dir, split_name, 'CAM_FRONT_RGB', f'front_{i:06d}.png')
            shutil.copy2(src_img_2, dst_img_2)
            
            # 右相机
            src_img_3 = os.path.join(image_3_dir, filename)
            dst_img_3 = os.path.join(output_dir, split_name, 'CAM_RIGHT_RGB', f'right_{i:06d}.png')
            shutil.copy2(src_img_3, dst_img_3)
            
            # 创建虚拟的左相机和后相机图像（基于前相机）
            img_2 = cv2.imread(src_img_2)
            img_3 = cv2.imread(src_img_3)
            
            # 左相机（翻转右相机）
            left_img = cv2.flip(img_3, 1)
            dst_left = os.path.join(output_dir, split_name, 'CAM_LEFT_RGB', f'left_{i:06d}.png')
            cv2.imwrite(dst_left, left_img)
            
            # 后相机（翻转前相机）
            rear_img = cv2.flip(img_2, 1)
            dst_rear = os.path.join(output_dir, split_name, 'CAM_REAR_RGB', f'rear_{i:06d}.png')
            cv2.imwrite(dst_rear, rear_img)
            
            # 创建BEV分割图像（基于标注）
            bev_img = create_bev_from_annotations(objects)
            dst_bev = os.path.join(output_dir, split_name, 'bev_segmentation', f'bev_{i:06d}.png')
            cv2.imwrite(dst_bev, bev_img)
            
            # 创建标注文件
            annotation = create_annotation_file(calib_data, objects, i)
            dst_ann = os.path.join(output_dir, split_name, 'annotations', f'ann_{i:06d}.json')
            with open(dst_ann, 'w') as f:
                json.dump(annotation, f, indent=2)
    
    print("KITTI数据转换完成！")

def create_bev_from_annotations(objects):
    """从标注创建BEV分割图像"""
    bev_size = (400, 400)
    bev_img = np.zeros((bev_size[1], bev_size[0], 3), dtype=np.uint8)
    bev_img[:] = (50, 50, 50)  # 深灰色背景
    
    # 绘制道路区域
    cv2.rectangle(bev_img, (50, 150), (350, 250), (128, 128, 128), -1)
    
    # 绘制相机位置
    cv2.circle(bev_img, (bev_size[0]//2, bev_size[1]//2), 5, (0, 255, 0), -1)
    
    # 绘制3D对象
    for obj in objects:
        if obj['type'] in ['Car', 'Van', 'Truck']:
            # 车辆
            x, y, z = obj['location']
            # 将3D坐标映射到BEV图像
            bev_x = int((x + 10) * 20)  # 映射到0-400
            bev_y = int((y + 10) * 20)
            
            if 0 <= bev_x < bev_size[0] and 0 <= bev_y < bev_size[1]:
                cv2.circle(bev_img, (bev_x, bev_y), 8, (255, 0, 0), -1)
        
        elif obj['type'] in ['Pedestrian', 'Person_sitting']:
            # 行人
            x, y, z = obj['location']
            bev_x = int((x + 10) * 20)
            bev_y = int((y + 10) * 20)
            
            if 0 <= bev_x < bev_size[0] and 0 <= bev_y < bev_size[1]:
                cv2.circle(bev_img, (bev_x, bev_y), 5, (0, 255, 0), -1)
    
    return bev_img

def create_annotation_file(calib_data, objects, index):
    """创建项目格式的标注文件"""
    # KITTI相机参数
    P_rect_02 = calib_data['P_rect_02'].reshape(3, 4)  # 左相机
    P_rect_03 = calib_data['P_rect_03'].reshape(3, 4)  # 右相机
    
    # 相机内参
    K_02 = P_rect_02[:3, :3]
    K_03 = P_rect_03[:3, :3]
    
    # 创建标注
    annotation = {
        "timestamp": 1234567890000 + index * 100000,
        "cams": {
            "CAM_FRONT_RGB": {
                "data_path": f"./CAM_FRONT_RGB/front_{index:06d}.png",
                "sensor2lidar_rotation": [1.0, 0.0, 0.0, 0.0],
                "sensor2lidar_translation": [0.0, 0.0, 0.0],
                "cam_intrinsic": K_02.tolist()
            },
            "CAM_LEFT_RGB": {
                "data_path": f"./CAM_LEFT_RGB/left_{index:06d}.png",
                "sensor2lidar_rotation": [0.707, 0.0, 0.707, 0.0],
                "sensor2lidar_translation": [0.0, -0.54, 0.0],  # KITTI基线
                "cam_intrinsic": K_03.tolist()
            },
            "CAM_RIGHT_RGB": {
                "data_path": f"./CAM_RIGHT_RGB/right_{index:06d}.png",
                "sensor2lidar_rotation": [0.707, 0.0, -0.707, 0.0],
                "sensor2lidar_translation": [0.0, 0.54, 0.0],  # KITTI基线
                "cam_intrinsic": K_03.tolist()
            },
            "CAM_REAR_RGB": {
                "data_path": f"./CAM_REAR_RGB/rear_{index:06d}.png",
                "sensor2lidar_rotation": [0.0, 0.0, 1.0, 0.0],
                "sensor2lidar_translation": [-2.0, 0.0, 0.0],
                "cam_intrinsic": K_02.tolist()
            },
            "CAM_BEV_SEGMENTATION": {
                "data_path": f"./bev_segmentation/bev_{index:06d}.png"
            }
        },
        "ann_infos": {
            "gt_names": [],
            "gt_boxes_3d": []
        }
    }
    
    # 添加3D标注
    for obj in objects:
        if obj['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting']:
            x, y, z = obj['location']
            l, h, w = obj['dimensions']  # KITTI格式: length, height, width
            rotation_y = obj['rotation_y']
            
            # 转换为项目格式: [x, y, z, l, w, h, yaw]
            gt_box = [x, y, z, l, w, h, rotation_y]
            
            annotation["ann_infos"]["gt_names"].append(obj['type'].lower())
            annotation["ann_infos"]["gt_boxes_3d"].append(gt_box)
    
    return annotation

def main():
    parser = argparse.ArgumentParser(description='下载和处理KITTI数据集')
    parser.add_argument('--download', action='store_true', 
                       help='下载KITTI数据集')
    parser.add_argument('--convert', action='store_true',
                       help='转换为项目格式')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='处理的样本数量')
    parser.add_argument('--output_dir', type=str, default='data/offline_images',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.download:
        download_kitti_dataset()
    
    if args.convert:
        kitti_dir = "data/kitti_raw"
        if not os.path.exists(kitti_dir):
            print("请先下载KITTI数据集: python download_kitti_data.py --download")
            return
        
        convert_kitti_to_project_format(kitti_dir, args.output_dir, args.num_samples)
    
    if not args.download and not args.convert:
        # 默认执行完整流程
        print("执行完整的KITTI数据处理流程...")
        download_kitti_dataset()
        convert_kitti_to_project_format("data/kitti_raw", args.output_dir, args.num_samples)
    
    print("\n数据处理完成！")
    print(f"数据位置: {args.output_dir}")
    print("\n使用方法:")
    print("1. 运行推理: python offline_inference.py --data_root data/offline_images/test --vis ./output")
    print("2. 开始训练: python train_simple.py model_interface/config/train_config.py --gpus 1")
    print("3. 验证数据: python validate_data.py --data_root data/offline_images/test --sample_idx 0")

if __name__ == '__main__':
    main() 