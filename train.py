#!/usr/bin/env python3
"""
BEV 3D感知模型训练脚本
基于MMDet3D框架
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
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

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

    # 创建数据加载器
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for dataset in datasets
    ]

    # 设置优化器
    optimizer = build_optimizer(model, cfg.optimizer)

    # 设置学习率调度器
    lr_config = cfg.get('lr_config', None)
    if lr_config is not None:
        lr_config = copy.deepcopy(lr_config)
        lr_config['step'] = [step * len(data_loaders[0]) for step in lr_config['step']]
        runner.register_lr_hook(lr_config)

    # 设置检查点配置
    checkpoint_config = cfg.get('checkpoint_config', None)
    if checkpoint_config is not None:
        checkpoint_config = copy.deepcopy(checkpoint_config)
        checkpoint_config['interval'] = checkpoint_config['interval'] * len(data_loaders[0])

    # 设置日志配置
    log_config = cfg.get('log_config', None)
    if log_config is not None:
        log_config = copy.deepcopy(log_config)
        log_config['interval'] = log_config['interval'] * len(data_loaders[0])

    # 设置评估配置
    evaluation = cfg.get('evaluation', None)
    if evaluation is not None:
        evaluation = copy.deepcopy(evaluation)
        evaluation['interval'] = evaluation['interval'] * len(data_loaders[0])

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