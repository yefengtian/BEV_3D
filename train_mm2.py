#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用训练脚本（MMCV 2.x + MMEngine + 官方 mmdet3d）
- 不使用项目内自带的 mmdet3d
- 兼容新/旧两种配置风格（旧风格做最小转换）
"""

import os
import sys
import copy
import glob
import argparse
import warnings
from typing import Tuple, Dict, Any

# ========= 仅使用官方包 =========
import torch

try:
    # 新栈
    from mmengine.config import Config, DictAction
    from mmengine.runner import Runner, set_random_seed
    from mmengine.utils import mkdir_or_exist
    from mmengine.logging import print_log
    from mmengine.model.utils import revert_sync_batchnorm
except Exception as e:
    raise RuntimeError(
        f"[FATAL] 需要 MMEngine / MMCV 2.x 新栈，请确认环境。原始错误：{e}"
    )

# mmdet3d 的注册与 registry
try:
    # 新版本常见用法
    from mmdet3d.utils import register_all_modules  # type: ignore
    register_all_modules()
except Exception:
    # 旧一些的版本可能用 init_default_scope 或别名
    try:
        from mmdet3d.utils import init_default_scope  # type: ignore
        init_default_scope('mmdet3d')
    except Exception:
        warnings.warn(
            "未找到 register_all_modules/init_default_scope，若后续构建失败，请升级 mmdet3d。"
        )

# 导入项目内的自定义模块以注册自定义组件
try:
    import sys
    import os
    # 将项目根目录添加到 Python 路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 导入项目内的 mmdet3d 模块以注册自定义组件
    from model_interface.mmdet3d.models import backbones, necks, dense_heads, detectors
    from model_interface.mmdet3d.models.losses import focal_loss
    print("成功导入项目内自定义组件")
except Exception as e:
    print(f"导入项目内自定义组件时出错: {e}")
    warnings.warn("无法导入项目内自定义组件，某些自定义模型可能无法使用")

# ============== 小工具 ==============

def find_latest_ckpt(work_dir: str):
    """在 work_dir 下寻找最新的 *.pth（按修改时间）"""
    if not work_dir or not os.path.isdir(work_dir):
        return None
    candidates = glob.glob(os.path.join(work_dir, "*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def freeze_modules(model: torch.nn.Module, module_names):
    """按模块名前缀冻结参数（例如：['backbone', 'neck.img_backbone']）"""
    if not module_names:
        return 0, 0
    to_freeze = [m.strip() for m in module_names if m.strip()]
    total, frozen = 0, 0
    for name, p in model.named_parameters():
        total += 1
        if any(name.startswith(pref) for pref in to_freeze):
            p.requires_grad = False
            frozen += 1
    return frozen, total


def is_v2_style_cfg(cfg: Config) -> bool:
    """判定是否为 OpenMMLab 2.x 新风格配置（含 train_dataloader/optim_wrapper/train_cfg）"""
    has_loader = ('train_dataloader' in cfg) or ('val_dataloader' in cfg) or ('test_dataloader' in cfg)
    has_optim_wrapper = 'optim_wrapper' in cfg
    has_train_cfg = 'train_cfg' in cfg or 'train_cfg' in getattr(cfg, '_cfg_dict', {})
    return bool(has_loader and has_optim_wrapper and has_train_cfg)


def _guess_bs_workers_from_old(cfg) -> Tuple[int, int]:
    """从旧 cfg.data.* 猜测 batch_size 与 num_workers"""
    bs = 2
    nw = 2
    data = getattr(cfg, 'data', None)
    if data:
        bs = int(getattr(data, 'samples_per_gpu', bs) or bs)
        nw = int(getattr(data, 'workers_per_gpu', nw) or nw)
    return bs, nw


def convert_old_cfg_to_v2(cfg: Config) -> Config:
    """
    将旧风格 cfg（含 cfg.data/optimizer/runner 等）最小转换为 2.x 风格：
    - data.train/val → train_dataloader/val_dataloader
    - optimizer → optim_wrapper
    - runner.max_epochs / total_epochs → train_cfg.max_epochs
    - work_dir 保留
    注意：这是最小可用映射，若有高度自定义 pipeline/sampler，请自行在 cfg 中完善。
    """
    if is_v2_style_cfg(cfg):
        return cfg  # 已是新风格

    if 'data' not in cfg:
        raise ValueError("旧风格转换失败：cfg 中没有 data 字段，无法构造 dataloader。")

    new_cfg = copy.deepcopy(cfg)

    # 1) dataloaders
    bs, nw = _guess_bs_workers_from_old(cfg)

    # 默认采样器与 collate（可按需调整）
    default_train_sampler = dict(type='DefaultSampler', shuffle=True)
    default_val_sampler = dict(type='DefaultSampler', shuffle=False)

    # 兼容一些旧字段命名：cfg.data.train / cfg.data.val 可能是 dict 或 ConfigDict
    train_ds = copy.deepcopy(cfg.data.train)
    val_ds = copy.deepcopy(getattr(cfg.data, 'val', None))

    new_cfg.train_dataloader = dict(
        batch_size=bs,
        num_workers=nw,
        sampler=default_train_sampler,
        dataset=train_ds,
        persistent_workers=True
    )
    if val_ds is not None:
        new_cfg.val_dataloader = dict(
            batch_size=max(1, bs),  # 验证通常 batch=1 或小一些
            num_workers=max(1, nw // 2),
            sampler=default_val_sampler,
            dataset=val_ds,
            persistent_workers=True
        )

    # 2) 优化器包装
    if 'optim_wrapper' not in new_cfg:
        if 'optimizer' in new_cfg:
            new_cfg.optim_wrapper = dict(optimizer=new_cfg.optimizer)
        else:
            raise ValueError("旧风格 cfg 缺少 optimizer，无法自动构造 optim_wrapper。")

    # 3) 训练循环（epoch-based）
    max_epochs = None
    runner_cfg = getattr(cfg, 'runner', None)
    if runner_cfg:
        max_epochs = runner_cfg.get('max_epochs', runner_cfg.get('total_epochs', None))
    if max_epochs is None:
        # 兜底：尝试从 lr_config 或自定义字段里拿；拿不到则给出默认 12
        max_epochs = int(getattr(cfg, 'max_epochs', 12))

    if 'train_cfg' not in new_cfg:
        new_cfg.train_cfg = dict(
            type='EpochBasedTrainLoop',
            max_epochs=max_epochs,
            val_interval=1  # 默认每个 epoch 验证
        )
    else:
        # 若已存在但没给 max_epochs，就补上
        new_cfg.train_cfg.setdefault('max_epochs', max_epochs)

    # 4) 验证/测试循环（可选）
    # new_cfg.setdefault('val_cfg', dict(type='ValLoop'))
    # new_cfg.setdefault('test_cfg', dict(type='TestLoop'))

    # 5) 参数调度（如果旧 cfg 里有 lr_config，可映射为 param_scheduler；这里给最小兜底）
    if 'param_scheduler' not in new_cfg:
        if 'lr_config' in new_cfg:
            # 复杂情况建议你在原 cfg 中添加 param_scheduler；这里不给武断转换
            print_log("检测到旧 lr_config，建议迁移为 param_scheduler。当前先不自动转换。", 'current')
        else:
            # 给个线性 warmup + multistep 的示例（可按需替换）
            new_cfg.param_scheduler = [
                dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
                dict(type='MultiStepLR', by_epoch=True, milestones=[max_epochs // 2, int(max_epochs * 0.8)], gamma=0.1)
            ]

    # 6) 默认可视化/钩子（可按需补充）
    new_cfg.setdefault('default_scope', 'mmdet3d')
    new_cfg.setdefault('default_hooks', dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True, max_keep_ckpts=3),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='Det3DVisualizationHook', draw=True, interval=0)
    ))

    # 7) 环境与日志
    new_cfg.setdefault('env_cfg', dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'),
    ))
    new_cfg.setdefault('vis_backends', [dict(type='LocalVisBackend')])
    new_cfg.setdefault('visualizer', dict(type='Det3DLocalVisualizer', vis_backends=new_cfg.vis_backends, name='visualizer'))

    # 8) launcher & 分布式参数（如果旧 cfg 有 dist_params）
    if 'launcher' in new_cfg:
        new_cfg.setdefault('dist_params', getattr(cfg, 'dist_params', dict(backend='nccl')))

    return new_cfg


# ============== CLI ==============

def parse_args():
    parser = argparse.ArgumentParser(description='Train a BEV 3D perception model (MMEngine/MCV2)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config (支持新/旧两种风格)')
    parser.add_argument('--work-dir', type=str, default=None, help='save logs & checkpoints here')
    parser.add_argument('--resume-from', type=str, default=None, help='resume runner state (optimizer/epoch/iters)')
    parser.add_argument('--auto-resume', action='store_true', help='auto locate latest *.pth in work_dir to resume')
    parser.add_argument('--load-from', type=str, default=None, help='load model weights only (finetune)')
    parser.add_argument('--ignore-missing-keys', action='store_true', help='strict=False when load-from')
    parser.add_argument('--freeze-modules', type=str, default='', help='comma separated prefixes, e.g. "backbone,neck.img_backbone"')
    parser.add_argument('--no-validate', action='store_true', help='do not run validation during training')
    parser.add_argument('--seed', type=int, default=5467, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='cudnn deterministic')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher (仅作为 cfg 透传)')
    parser.add_argument('--options', nargs='+', action=DictAction, help='override settings in config e.g., key=value')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg: Config = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # work_dir 处理
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.setdefault('work_dir', os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0]))

    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # 旧→新配置转换（如有必要）
    if not is_v2_style_cfg(cfg):
        print_log("检测到旧风格配置：尝试进行最小转换为 2.x 风格...", 'current')
        cfg = convert_old_cfg_to_v2(cfg)

    # 训练/验证开关
    # 在新风格里，validate 是通过 train_cfg/val_cfg、default_hooks.checkpoint/runner.loop 协同的。
    # 这里提供一个快速关闭验证的开关（把 val_interval 设为 0）。
    if args.no_validate:
        if 'train_cfg' in cfg and isinstance(cfg.train_cfg, dict):
            cfg.train_cfg['val_interval'] = 0

    # 随机种子
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.setdefault('randomness', dict(seed=args.seed, deterministic=args.deterministic))

    # launcher 透传（可选）
    cfg.setdefault('launcher', args.launcher)
    cfg.setdefault('log_level', 'INFO')

        # ---- 强制关闭验证时，删掉 val_* 以通过 MMEngine 校验 ----
    if args.no_validate:
        for k in ['val_dataloader', 'val_cfg', 'val_evaluator']:
            if k in cfg:
                cfg.pop(k)

    # 构建 Runner
    runner = Runner.from_cfg(cfg)

    # 可选：加载仅权重（finetune）
    if (args.load_from is not None) and (args.resume_from is None) and (not args.auto_resume):
        ckpt_path = args.load_from
        strict = not args.ignore_missing_keys
        print_log(f"[finetune] load weights only: {ckpt_path} (strict={strict})", 'current')
        # 直接让 runner 加载模型权重
        runner.load_checkpoint(ckpt_path, map_location='cpu', revise_keys=None)  # revise_keys 按需设置

    # 可选：自动续训（从最新 *.pth 恢复状态）
    resume_from = args.resume_from
    if args.auto_resume and resume_from is None:
        latest = find_latest_ckpt(cfg.work_dir)
        if latest:
            resume_from = latest
            print_log(f"[auto-resume] Found latest: {resume_from}", 'current')

    # 冻结模块
    if args.freeze_modules:
        prefixes = [p.strip() for p in args.freeze_modules.split(',') if p.strip()]
        model = runner.model
        # 若为 SyncBN，训练前可换回 BN 以便冻结更直观（可选）
        try:
            model = revert_sync_batchnorm(model)
            runner.model = model
        except Exception:
            pass
        frozen, total = freeze_modules(model, prefixes)
        print_log(f"[freeze] Frozen params: {frozen}/{total} (trainable: {total - frozen}) | prefixes={prefixes}", 'current')

    # 正式开跑
    if resume_from:
        print_log(f"[resume] resume from: {resume_from}", 'current')
        runner.resume(resume_from)  # 恢复 optimizer/epoch/iters 等
    runner.train()


if __name__ == '__main__':
    main()
