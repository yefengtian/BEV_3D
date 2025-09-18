#!/usr/bin/env python3
"""
BEV 3D感知模型训练脚本
基于MMDet3D框架
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'model_interface'))

import argparse
import torch
import mmcv
import copy
import glob
# from mmcv import Config
try:
    # 新栈（MMCV 2.x + MMEngine）
    from mmengine.config import Config, DictAction, ConfigDict
except Exception:
    # 旧栈（MMCV 1.x）
    from mmcv import Config
    from mmcv.utils import DictAction, ConfigDict


# from mmdet3d.utils import setup_multi_processes, compat_cfg

# 尝试导入 vendored 的工具（大概率会因版本断言失败），失败则用本地兜底
try:
    # 注意：这行会触发 vendored mmdet3d 的 __init__，在 mmcv==2.x 下会 AssertionError
    from mmdet3d.utils import setup_multi_processes as _setup_mp, compat_cfg as _compat_cfg  # noqa
    setup_multi_processes = _setup_mp
    compat_cfg = _compat_cfg
except Exception as e:
    warnings.warn(f"mmdet3d.utils 无法导入（老版本断言/依赖不兼容），使用本地兜底实现: {e}")

    def compat_cfg(cfg):
        """最小化兼容：如无特别需要，直接原样返回即可。
        若你的 cfg 里有 'opencv_num_threads'/'cudnn_benchmark' 等键，本地也会用到。
        """
        return cfg

    def setup_multi_processes(cfg):
        """最小版的多进程/线程设置，与 mmdet3d/utils 中的常见做法等价够用。"""
        # 限制 OpenCV/omp/mkl 线程数，避免 CPU 抢占
        try:
            import cv2  # noqa
            # 默认 0 关闭 OpenCV 线程；如 cfg 明确设置则以 cfg 为准
            num = int(cfg.get('opencv_num_threads', 0))
            try:
                cv2.setNumThreads(num)
            except Exception:
                pass
        except Exception:
            pass

        os.environ.setdefault('OMP_NUM_THREADS', str(cfg.get('omp_num_threads', 1)))
        os.environ.setdefault('MKL_NUM_THREADS', str(cfg.get('mkl_num_threads', 1)))

        # cuDNN benchmark 开关
        try:
            import torch
            torch.backends.cudnn.benchmark = bool(cfg.get('cudnn_benchmark', False))
        except Exception:
            pass
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets import CarlaDataset
from mmdet3d.datasets import build_dataloader
from mmdet3d.models import build_model

from mmcv.runner import load_checkpoint,init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.datasets import DATASETS
print(DATASETS.module_dict.keys())

os.environ

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
        return 0,0
    to_freeze = set([m.strip() for m in module_names if m.strip()])
    total, frozen = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if any(name.startswith(prefix) for prefix in to_freeze):
            param.requires_grad = False
            frozen += 1
    return frozen, total

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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a BEV 3D perception model')
    parser.add_argument("--config", type=str, default="/workspace/drWorkspace/BEVParkingOL/model_interface/config/freespace_occ2d_r50_depth.py")
    parser.add_argument('--work-dir', help='the dir to save logs and models',default = '/workspace/drWorkspace/BEVParkingOL/freespace_0915')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--load-from', help='load checkpoint weights for finetuning (仅加载权重，不恢复优化器与进度)')
    parser.add_argument('--ignore-missing-keys', action='store_true',help='load_from 时 strict=False，忽略缺失/不匹配的权重键')
    parser.add_argument('--auto-resume', action='store_true',help='若未显式指定 --resume-from，则在 work_dir 下自动寻找最新的 *.pth 进行续训')
    parser.add_argument('--freeze-modules', type=str, default='',help='以逗号分隔的模块名前缀列表，例如 "backbone,neck.img_backbone"')
    parser.add_argument('--no-validate', action='store_true',help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=5467, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


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

    # 创建输出目录
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # 初始化分布式训练
    if args.launcher == 'none' or args.gpus==1:
        distributed = False
        cfg.gpu_ids = [0]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        cfg.gpu_ids = list(range(args.gpus))


    # 自动缩放学习率
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

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

    # 续训
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


    # 设置优化器
    # optimizer = build_optimizer(model, cfg.optimizer)

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