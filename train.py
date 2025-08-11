#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV 3D感知模型训练脚本（最终版）
- 支持：resume / auto-resume / load_from（strict 可选）/ 冻结模块
- 支持：VSCode + torch.distributed.run 的单机多卡调试
"""

import os
import sys
import glob
import argparse
import copy

import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint, build_optimizer, init_dist, get_dist_info

from mmdet3d.utils import setup_multi_processes, compat_cfg
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# 保证自定义数据集/管道已注册（按你的工程结构）
import model_interface.mmdet3d.datasets.pipelines.loading  # noqa: F401
import model_interface.mmdet3d.datasets.carla_dataset      # noqa: F401

from mmdet3d.datasets import DATASETS, PIPELINES


# ---------------- 工具函数 ----------------
def find_latest_ckpt(work_dir: str):
    """在 work_dir 下寻找最新的 *.pth（按修改时间降序）"""
    if not work_dir or not os.path.isdir(work_dir):
        return None
    cands = glob.glob(os.path.join(work_dir, "*.pth"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def freeze_modules(model, module_names):
    """按参数名前缀冻结参数（e.g. ['backbone', 'neck.img_backbone']）"""
    if not module_names:
        return 0, 0
    to_freeze = set([m.strip() for m in module_names if m.strip()])
    total, frozen = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if any(name.startswith(prefix) for prefix in to_freeze):
            param.requires_grad = False
            frozen += 1
    return frozen, total


def ensure_cfg_defaults(cfg):
    """确保一些关键默认项存在"""
    if not hasattr(cfg, "work_dir") or cfg.work_dir is None:
        cfg.work_dir = os.path.join("./work_dirs",
                                    os.path.splitext(os.path.basename(cfg.filename))[0])
    if not hasattr(cfg, "dist_params") or not cfg.dist_params:
        cfg.dist_params = dict(backend="nccl")
    return cfg


# ---------------- argparse ----------------
class DictAction(argparse.Action):
    """支持 KEY=VALUE 的 dict 解析"""
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(DictAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BEV 3D perception model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="checkpoint to resume from（恢复训练进度）")
    # 仅加载权重微调，不恢复进度
    parser.add_argument("--load-from", help="checkpoint to load weights for finetuning")
    parser.add_argument("--ignore-missing-keys", action="store_true",
                        help="load_from 时 strict=False，忽略不匹配的权重键")
    # 自动寻找最新权重续训
    parser.add_argument("--auto-resume", action="store_true",
                        help="若未显式 --resume-from，则在 work_dir 下寻找最新 *.pth 续训")
    # 冻结部分模块
    parser.add_argument("--freeze-modules", type=str, default="",
                        help='以逗号分隔模块名前缀，例如 "backbone,neck.img_backbone"')
    parser.add_argument("--no-validate", action="store_true",
                        help="whether not to evaluate during training")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--seed", type=int, default=32, help="random seed")
    parser.add_argument("--deterministic", action="store_true",
                        help="use deterministic CUDNN")
    parser.add_argument("--options", nargs="+", action=DictAction, help="cfg overrides")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"],
                        default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--autoscale-lr", action="store_true",
                        help="automatically scale lr with number of gpus")
    args = parser.parse_args()
    # 补 LOCAL_RANK（便于 VSCode 单进程调试）
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


# ---------------- 打印控制（仅 rank0 打印） ----------------
_builtin_print = print
def rank0_print(*a, **kw):
    if int(os.environ.get("RANK", "0")) == 0:
        _builtin_print(*a, **kw)


# ---------------- 主流程 ----------------
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    cfg = ensure_cfg_defaults(cfg)

    # 覆盖 work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 随机性
    if args.seed is not None:
        cfg.seed = args.seed
        if args.deterministic:
            cfg.deterministic = True

    # GPU 数量（MMDP 不建议；DDP 用 torchrun 控制进程数）
    if args.gpus is not None:
        cfg.gpu_ids = range(args.gpus)

    # 自动缩放学习率
    if args.autoscale_lr and "optimizer" in cfg and "lr" in cfg.optimizer:
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(getattr(cfg, "gpu_ids", [0])) / 8

    # 多进程 & 输出目录
    setup_multi_processes(cfg)
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # -------- 分布式初始化（VSCode + torch.distributed.run） --------
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        if args.launcher == "pytorch":
            # 检查 torchrun 注入的环境变量
            needed = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
            missing = [k for k in needed if k not in os.environ]
            if missing:
                raise RuntimeError(
                    f"[DDP] 缺少环境变量: {missing}\n"
                    "请用 VSCode 的 'module: torch.distributed.run' 启动，或命令行 torchrun。"
                )
            # 绑定本地卡
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        init_dist(args.launcher, **cfg.dist_params)

    # -------- 构建数据集 --------
    rank0_print("Available datasets:", list(DATASETS.module_dict.keys()))
    rank0_print("Available pipelines:", list(PIPELINES.module_dict.keys()))

    datasets = [build_dataset(cfg.data.train)]
    if len(getattr(cfg, "workflow", [])) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # -------- 构建模型 --------
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.CLASSES = datasets[0].CLASSES

    # -------- 续训练 / 微调 逻辑 --------
    resume_from = args.resume_from
    if args.auto_resume and not resume_from:
        latest = find_latest_ckpt(cfg.work_dir)
        if latest:
            resume_from = latest
            rank0_print(f"[auto-resume] Found latest checkpoint: {resume_from}")

    if resume_from:
        cfg.resume_from = resume_from
        rank0_print(f"[resume] Will resume training from: {cfg.resume_from}")
    elif args.load_from:
        strict = not args.ignore_missing_keys
        ckpt_path = args.load_from
        rank0_print(f"[finetune] Load weights from: {ckpt_path} (strict={strict})")
        _ = load_checkpoint(model, ckpt_path, map_location="cpu", strict=strict)
        cfg.load_from = ckpt_path  # 记录到 cfg 便于复现

    # -------- 冻结模块（可选） --------
    if args.freeze_modules:
        prefixes = [p.strip() for p in args.freeze_modules.split(",") if p.strip()]
        frozen, total = freeze_modules(model, prefixes)
        trainable = total - frozen
        rank0_print(f"[freeze] Frozen params: {frozen} / {total} (trainable: {trainable}). Prefixes={prefixes}")

    # （可选）构建优化器：仅用于打印 param_groups；真正训练时 MMCV 会在 runner 内部再 build 一次
    if hasattr(cfg, "optimizer"):
        opt_tmp = build_optimizer(model, cfg.optimizer)
        sizes = [len(g["params"]) for g in opt_tmp.param_groups]
        rank0_print(f"[optimizer] param_groups={len(sizes)} sizes={sizes}")
        del opt_tmp

    # -------- 启动训练 --------
    train_model(
        model=model,
        dataset=datasets,
        cfg=cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=None,
        meta=None,
    )

    # 训练结束
    if distributed:
        rank, world = get_dist_info()
        if rank == 0:
            print("[done] training finished.")
    else:
        print("[done] training finished (single process).")


if __name__ == "__main__":
    main()
