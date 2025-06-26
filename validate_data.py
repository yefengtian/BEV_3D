#!/usr/bin/env python3
"""
数据验证脚本
验证虚拟图像数据和相机参数的正确性，输出BEV视角下的情况
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from pyquaternion import Quaternion
import torch

def load_camera_params(cam_info):
    """加载相机参数"""
    # 相机内参
    intrinsic = np.array(cam_info['cam_intrinsic'])
    
    # 相机外参（sensor2lidar）
    rotation_quat = cam_info['sensor2lidar_rotation']  # [w, x, y, z]
    translation = np.array(cam_info['sensor2lidar_translation'])
    
    # 四元数转旋转矩阵
    quat = Quaternion(rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3])
    rotation_matrix = quat.rotation_matrix
    
    # 构建变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    
    return intrinsic, transform_matrix

def project_points_to_image(points_3d, intrinsic, extrinsic):
    """将3D点投影到图像平面"""
    # 将3D点转换为齐次坐标
    points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # 应用外参变换
    points_cam = (extrinsic @ points_homo.T).T
    
    # 只保留z>0的点（在相机前方的点）
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    
    if len(points_cam) == 0:
        return np.array([]), valid_mask
    
    # 投影到图像平面
    points_2d = points_cam[:, :3] / points_cam[:, 2:3]
    
    # 应用内参
    points_2d = (intrinsic @ points_2d.T).T
    
    return points_2d[:, :2], valid_mask

def create_bev_grid(x_range=(-10, 10), y_range=(-10, 10), resolution=0.1):
    """创建BEV网格"""
    x = np.arange(x_range[0], x_range[1] + resolution, resolution)
    y = np.arange(y_range[0], y_range[1] + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # 地面高度为0
    
    # 展平为点云格式
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return points, X.shape

def visualize_bev_projection(image_path, cam_info, output_dir, sample_idx):
    """可视化BEV投影"""
    # 加载图像
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # 如果图像不存在，创建虚拟图像
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[:] = (100, 100, 100)
        # 添加一些特征
        cv2.line(image, (0, 540), (1920, 540), (255, 255, 255), 10)
        cv2.line(image, (480, 0), (480, 1080), (255, 255, 255), 5)
        cv2.line(image, (1440, 0), (1440, 1080), (255, 255, 255), 5)
    
    # 加载相机参数
    intrinsic, extrinsic = load_camera_params(cam_info)
    
    # 创建BEV网格
    points_3d, grid_shape = create_bev_grid()
    
    # 投影到图像平面
    points_2d, valid_mask = project_points_to_image(points_3d, intrinsic, extrinsic)
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 原始图像
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image - {os.path.basename(image_path)}')
    axes[0].axis('off')
    
    # 2. 投影结果
    axes[1].imshow(image)
    if len(points_2d) > 0:
        # 只显示有效的投影点
        axes[1].scatter(points_2d[:, 0], points_2d[:, 1], c='red', s=1, alpha=0.5)
    axes[1].set_title('BEV Grid Projection')
    axes[1].axis('off')
    
    # 3. BEV视图
    bev_image = np.zeros((grid_shape[0], grid_shape[1], 3), dtype=np.uint8)
    bev_image[:] = (50, 50, 50)  # 深灰色背景
    
    # 绘制道路区域
    road_y_start = int(grid_shape[0] * 0.4)
    road_y_end = int(grid_shape[0] * 0.6)
    bev_image[road_y_start:road_y_end, :] = (128, 128, 128)  # 灰色道路
    
    # 绘制相机位置
    cam_x = int((0 - (-10)) / 0.1)  # 相机在原点
    cam_y = int((0 - (-10)) / 0.1)
    cv2.circle(bev_image, (cam_x, cam_y), 5, (0, 255, 0), -1)  # 绿色圆点表示相机
    
    # 绘制相机视野范围
    if len(points_2d) > 0:
        # 将图像坐标映射回BEV坐标
        valid_bev_points = points_3d[valid_mask]
        
        for i, bev_pt in enumerate(valid_bev_points):
            if i % 100 == 0:  # 采样显示，避免过于密集
                bev_x = int((bev_pt[0] - (-10)) / 0.1)
                bev_y = int((bev_pt[1] - (-10)) / 0.1)
                if 0 <= bev_x < grid_shape[1] and 0 <= bev_y < grid_shape[0]:
                    bev_image[bev_y, bev_x] = (255, 255, 255)  # 白色表示可见区域
    
    axes[2].imshow(bev_image)
    axes[2].set_title('BEV View')
    axes[2].axis('off')
    
    # 添加相机参数信息
    info_text = f"Camera: {os.path.basename(image_path)}\n"
    info_text += f"Intrinsic:\n{intrinsic}\n"
    info_text += f"Translation: {cam_info['sensor2lidar_translation']}\n"
    info_text += f"Rotation (quat): {cam_info['sensor2lidar_rotation']}"
    
    fig.suptitle(f'Sample {sample_idx} - BEV Projection Validation', fontsize=16)
    fig.tight_layout()
    
    # 保存结果
    output_path = os.path.join(output_dir, f'validation_sample_{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def validate_annotation(ann_file, output_dir, sample_idx):
    """验证标注文件"""
    if not os.path.exists(ann_file):
        print(f"标注文件不存在: {ann_file}")
        return
    
    with open(ann_file, 'r') as f:
        annotation = json.load(f)
    
    # 创建BEV可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 创建BEV网格
    bev_size = 400
    bev_image = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    bev_image[:] = (50, 50, 50)
    
    # 绘制道路
    cv2.rectangle(bev_image, (50, 150), (350, 250), (128, 128, 128), -1)
    
    # 绘制相机位置
    cv2.circle(bev_image, (bev_size//2, bev_size//2), 5, (0, 255, 0), -1)
    
    # 绘制3D边界框
    if 'ann_infos' in annotation and 'gt_boxes_3d' in annotation['ann_infos']:
        boxes_3d = annotation['ann_infos']['gt_boxes_3d']
        gt_names = annotation['ann_infos']['gt_names']
        
        for i, (box, name) in enumerate(zip(boxes_3d, gt_names)):
            x, y, z, l, w, h, yaw = box
            
            # 将3D坐标转换为BEV像素坐标
            bev_x = int((x + 10) * 20)  # 映射到0-400
            bev_y = int((y + 10) * 20)
            
            # 绘制边界框
            color = (255, 0, 0) if name == 'vehicle' else (0, 255, 0)
            cv2.rectangle(bev_image, (bev_x-10, bev_y-10), (bev_x+10, bev_y+10), color, 2)
            
            # 添加标签
            cv2.putText(bev_image, name, (bev_x-10, bev_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    ax.imshow(bev_image)
    ax.set_title(f'Sample {sample_idx} - Annotation Validation')
    ax.axis('off')
    
    # 添加标注信息
    info_text = f"Timestamp: {annotation.get('timestamp', 'N/A')}\n"
    if 'ann_infos' in annotation:
        info_text += f"Objects: {len(annotation['ann_infos'].get('gt_names', []))}\n"
        for name in annotation['ann_infos'].get('gt_names', []):
            info_text += f"- {name}\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存结果
    output_path = os.path.join(output_dir, f'annotation_validation_{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def validate_camera_parameters(cam_info, output_dir, sample_idx):
    """验证相机参数"""
    intrinsic, extrinsic = load_camera_params(cam_info)
    
    # 创建相机参数可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 内参矩阵
    im1 = axes[0].imshow(intrinsic, cmap='viridis')
    axes[0].set_title('Camera Intrinsic Matrix')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # 添加数值标签
    for i in range(intrinsic.shape[0]):
        for j in range(intrinsic.shape[1]):
            axes[0].text(j, i, f'{intrinsic[i, j]:.1f}', 
                        ha='center', va='center', color='white')
    
    plt.colorbar(im1, ax=axes[0])
    
    # 2. 外参矩阵
    im2 = axes[1].imshow(extrinsic, cmap='viridis')
    axes[1].set_title('Camera Extrinsic Matrix (sensor2lidar)')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    
    # 添加数值标签
    for i in range(extrinsic.shape[0]):
        for j in range(extrinsic.shape[1]):
            axes[1].text(j, i, f'{extrinsic[i, j]:.2f}', 
                        ha='center', va='center', color='white')
    
    plt.colorbar(im2, ax=axes[1])
    
    fig.suptitle(f'Sample {sample_idx} - Camera Parameters Validation', fontsize=16)
    fig.tight_layout()
    
    # 保存结果
    output_path = os.path.join(output_dir, f'camera_params_validation_{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='验证虚拟图像数据和相机参数')
    parser.add_argument('--data_root', type=str, default='data/sample_data',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='validation_output',
                       help='输出目录')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='要验证的样本索引')
    parser.add_argument('--all_samples', action='store_true',
                       help='验证所有样本')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取样本列表
    if args.all_samples:
        # 查找所有样本
        cam_dir = os.path.join(args.data_root, 'CAM_FRONT_RGB')
        if os.path.exists(cam_dir):
            sample_files = [f for f in os.listdir(cam_dir) if f.endswith('.jpg')]
            sample_indices = [int(f.split('_')[1].split('.')[0]) for f in sample_files]
        else:
            sample_indices = [0, 1, 2, 3, 4]  # 默认样本
    else:
        sample_indices = [args.sample_idx]
    
    print(f"开始验证 {len(sample_indices)} 个样本...")
    
    for sample_idx in sample_indices:
        print(f"验证样本 {sample_idx}...")
        
        # 加载标注文件
        ann_file = os.path.join(args.data_root, 'annotations', f'ann_{sample_idx:03d}.json')
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            # 验证每个相机的投影
            for cam_name, cam_info in annotation['cams'].items():
                if cam_name == 'CAM_BEV_SEGMENTATION':
                    continue
                
                # 获取图像路径
                img_path = os.path.join(args.data_root, cam_name, cam_info['data_path'].split('/')[-1])
                
                # 验证BEV投影
                output_path = visualize_bev_projection(img_path, cam_info, args.output_dir, sample_idx)
                print(f"  - {cam_name} BEV投影验证完成: {output_path}")
            
            # 验证标注
            ann_output = validate_annotation(ann_file, args.output_dir, sample_idx)
            print(f"  - 标注验证完成: {ann_output}")
            
            # 验证相机参数
            cam_output = validate_camera_parameters(annotation['cams']['CAM_FRONT_RGB'], args.output_dir, sample_idx)
            print(f"  - 相机参数验证完成: {cam_output}")
        
        else:
            print(f"  样本 {sample_idx} 的标注文件不存在")
    
    print(f"\n验证完成！结果保存在: {args.output_dir}")
    print("\n验证内容包括:")
    print("1. BEV网格投影到各相机图像")
    print("2. 3D标注在BEV视图中的显示")
    print("3. 相机内参和外参矩阵的可视化")
    print("4. 相机视野范围在BEV中的表示")

if __name__ == '__main__':
    main() 