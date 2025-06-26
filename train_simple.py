#!/usr/bin/env python3
"""
简化的BEV 3D感知模型训练脚本
使用MMDet3D标准训练API
"""

import os
import sys
import argparse
import torch
import mmcv
from mmcv import Config
from mmdet.utils import setup_multi_processes, compat_cfg
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a BEV 3D perception model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)

    # 设置多进程
    setup_multi_processes(cfg)

    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # 设置随机种子
    if args.seed is not None:
        cfg.seed = args.seed

    # 设置GPU数量
    if args.gpus is not None:
        cfg.gpu_ids = range(args.gpus)

    # 创建输出目录
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # 设置恢复训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    print(f"Starting training with config: {args.config}")
    print(f"Work directory: {cfg.work_dir}")
    print(f"GPU IDs: {cfg.gpu_ids}")

    # 开始训练
    train_model(
        cfg,
        distributed=False,
        validate=True,
        timestamp=None,
        meta=None)

if __name__ == '__main__':
    main() 