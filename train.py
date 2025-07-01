#!/usr/bin/env python3
"""
BEV 3D感知模型训练脚本
基于MMDet3D框架
"""

import os
import sys
import argparse
import copy
import torch
import mmcv
from mmcv import Config
from mmdet.utils import setup_multi_processes, compat_cfg
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint, save_checkpoint, build_optimizer
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist

# 导入自定义数据集
import dataset.carla_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train a BEV 3D perception model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

class DictAction(argparse.Action):
    """argparse action to split an argument into KEY=VALUE form on the first =
    and append to a dictionary. List options can be passed as comma separated
    values, i.e 'KEY=V1,V2,V3', or with explicit brackets, i.e. 'KEY=[V1,V2,V3]'.
    It also supports nested brackets to build list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(DictAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = val
        setattr(namespace, self.dest, options)

def main():
    args = parse_args()

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
        if args.deterministic:
            cfg.deterministic = True

    # 设置GPU数量
    if args.gpus is not None:
        cfg.gpu_ids = range(args.gpus)

    # 自动缩放学习率
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # 确保数据配置参数存在
    if not hasattr(cfg, 'samples_per_gpu'):
        cfg.samples_per_gpu = 4
    if not hasattr(cfg, 'workers_per_gpu'):
        cfg.workers_per_gpu = 4

    # 创建输出目录
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # 初始化分布式训练
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 创建数据集
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # 创建模型
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # 添加数据集到模型
    model.CLASSES = datasets[0].CLASSES

    # 设置优化器
    optimizer = build_optimizer(model, cfg.optimizer)

    # 开始训练
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=None,
        meta=None)

if __name__ == '__main__':
    main() 