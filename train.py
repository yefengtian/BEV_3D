#!/usr/bin/env python3
"""
BEV 3D感知模型训练脚本（含续训练/微调支持）
基于 MMDet3D 框架
"""

import os
import sys
import argparse
import copy
import glob
import torch
import mmcv
from mmcv import Config
from mmdet3d.utils import setup_multi_processes, compat_cfg
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint, build_optimizer, init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# 导入自定义组件以确保注册
import model_interface.mmdet3d.datasets.pipelines.loading  # noqa: F401
import model_interface.mmdet3d.datasets.carla_dataset      # noqa: F401

from mmdet3d.datasets import DATASETS, PIPELINES
print("Available datasets:", DATASETS.module_dict.keys())
print("Available pipelines:", PIPELINES.module_dict.keys())

# ---------------- NEW: 工具函数 ----------------
def find_latest_ckpt(work_dir: str):
    """在 work_dir 下寻找最新的 *.pth（按修改时间）"""
    if not work_dir or not os.path.isdir(work_dir):
        return None
    candidates = glob.glob(os.path.join(work_dir, "*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def freeze_modules(model, module_names):
    """按模块名前缀冻结参数（例如：['backbone', 'neck.img_backbone']）"""
    if not module_names:
        return 0
    to_freeze = set([m.strip() for m in module_names if m.strip()])
    total, frozen = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if any(name.startswith(prefix) for prefix in to_freeze):
            param.requires_grad = False
            frozen += 1
    return frozen, total

# ---------------- argparse ----------------
class DictAction(argparse.Action):
    """支持 KEY=VALUE 的 dict 解析"""
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(DictAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = val
        setattr(namespace, self.dest, options)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a BEV 3D perception model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from (恢复训练进度)')
    # ---------------- NEW: 仅加载权重微调，不恢复进度 ----------------
    parser.add_argument('--load-from', help='load checkpoint weights for finetuning (仅加载权重，不恢复优化器与进度)')
    parser.add_argument('--ignore-missing-keys', action='store_true',
                        help='load_from 时 strict=False，忽略缺失/不匹配的权重键')
    # ---------------- NEW: 自动寻找最新权重续训 ----------------
    parser.add_argument('--auto-resume', action='store_true',
                        help='若未显式指定 --resume-from，则在 work_dir 下自动寻找最新的 *.pth 进行续训')
    # ---------------- NEW: 冻结部分模块 ----------------
    parser.add_argument('--freeze-modules', type=str, default='',
                        help='以逗号分隔的模块名前缀列表，例如 "backbone,neck.img_backbone"')
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=32, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true',
                        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# ---------------- main ----------------
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)

    # 覆盖 work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # 随机种子/确定性
    if args.seed is not None:
        cfg.seed = args.seed
        if args.deterministic:
            cfg.deterministic = True

    # GPU 数量
    if args.gpus is not None:
        cfg.gpu_ids = range(args.gpus)

    # 自动缩放学习率
    if args.autoscale_lr and 'optimizer' in cfg and 'lr' in cfg.optimizer:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # 多进程设置 & 输出目录
    setup_multi_processes(cfg)
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # 分布式初始化
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    if len(getattr(cfg, 'workflow', [])) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # 构建模型
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    # ---------------- NEW: 处理 resume/load 逻辑 ----------------
    resume_from = args.resume_from
    if args.auto_resume and not resume_from:
        latest = find_latest_ckpt(cfg.work_dir)
        if latest:
            resume_from = latest
            print(f"[auto-resume] Found latest checkpoint: {resume_from}")

    # resume_from：恢复训练进度（包括优化器/epoch等），交由 runner 在 train_model 内处理
    if resume_from:
        cfg.resume_from = resume_from
        print(f"[resume] Will resume training from: {cfg.resume_from}")

    # load_from：仅加载权重用于微调（不恢复优化器/epoch）
    # 注意：如果同时给了 resume_from 与 load_from，以 resume_from 优先
    if (not resume_from) and args.load_from:
        strict = not args.ignore_missing-keys if False else None  # 占位避免编辑器误报
    if (not resume_from) and args.load_from:
        strict = not args.ignore_missing_keys
        ckpt_path = args.load_from
        print(f"[finetune] Loading weights from: {ckpt_path} (strict={strict})")
        _ = load_checkpoint(model, ckpt_path, map_location='cpu', strict=strict)
        # 也可以把 load_from 记录到 cfg 里（供日志/复现）
        cfg.load_from = ckpt_path

    # ---------------- NEW: 冻结指定模块 ----------------
    if args.freeze_modules:
        prefixes = [p.strip() for p in args.freeze_modules.split(',') if p.strip()]
        frozen, total = freeze_modules(model, prefixes)
        trainable = total - frozen
        print(f"[freeze] Frozen params: {frozen} / {total} (trainable: {trainable}). Prefixes={prefixes}")

    # 优化器
    optimizer = build_optimizer(model, cfg.optimizer)

    # 训练
    train_model(
        model=model,
        dataset=datasets,
        cfg=cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=None,
        meta=None
    )

if __name__ == '__main__':
    main()