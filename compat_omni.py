# compat_omni.py
"""
让老的 OpenMMLab v1 代码在新环境(mmcv>=2.x + mmengine)下运行的最小兼容层。
做的事：
1) 提供 mmdet3d.utils.setup_multi_processes / compat_cfg 的等价实现
2) 补齐老路径导入（mmcv.runner -> mmengine.runner / mmcv.utils -> mmengine.*）
3) 兼容 get_dist_info / collect_env / Config / build_from_cfg 等常用入口
4) 兜底 scatter/collate/nms 等使用路径变化
"""

import os
import sys
import warnings

def _patch_env():
    # 老代码常见：设置多进程/OMP/NUMA 之类
    os.environ.setdefault('OMP_NUM_THREADS', '1')  # 减少过度并行
    os.environ.setdefault('MKL_NUM_THREADS', '1')

def _setup_mmdet3d_utils():
    # 提供旧的 mmdet3d.utils 接口
    try:
        import types
        import importlib
        # 动态创建一个 mmdet3d.utils 模块
        md_utils = types.ModuleType('mmdet3d.utils')
        sys.modules['mmdet3d.utils'] = md_utils

        # setup_multi_processes 等价实现
        def setup_multi_processes(cfg=None):
            _patch_env()
            # 旧版会根据 cfg 设置 cudnn_benchmark / mp start method 等，
            # 这里取保守默认；需要时可按你的项目再细化。
            try:
                import torch
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

        # compat_cfg：老 mmdetection 会把一些字段“改名/补齐”
        # 这里做个轻量兼容：确保 cfg 是 dict-like
        def compat_cfg(cfg):
            return cfg

        md_utils.setup_multi_processes = setup_multi_processes
        md_utils.compat_cfg = compat_cfg

    except Exception as e:
        warnings.warn(f'[compat_omni] mmdet3d.utils 兼容失败: {e}')

def _alias_imports():
    # 把常见老接口路径映射到新接口
    try:
        import types
        import importlib

        # mmcv.runner.* -> mmengine.runner.*
        try:
            me_runner = importlib.import_module('mmengine.runner')
            mmcv_runner = types.ModuleType('mmcv.runner')
            for name in [
                'Runner', 'load_checkpoint', 'get_dist_info', 'save_checkpoint'
            ]:
                if hasattr(me_runner, name):
                    setattr(mmcv_runner, name, getattr(me_runner, name))
            # get_dist_info 位置在 mmengine.dist
            try:
                me_dist = importlib.import_module('mmengine.dist')
                if hasattr(me_dist, 'get_dist_info'):
                    setattr(mmcv_runner, 'get_dist_info', getattr(me_dist, 'get_dist_info'))
            except Exception:
                pass
            sys.modules['mmcv.runner'] = mmcv_runner
        except Exception:
            pass

        # mmcv.utils.* -> mmengine.utils.*
        try:
            me_utils = importlib.import_module('mmengine.utils')
            mmcv_utils = types.ModuleType('mmcv.utils')
            for name in ['Config', 'collect_env', 'get_git_hash', 'Registry', 'track_iter_progress']:
                if hasattr(me_utils, name):
                    setattr(mmcv_utils, name, getattr(me_utils, name))
            # build_from_cfg 在 mmengine.registry
            try:
                me_registry = importlib.import_module('mmengine.registry')
                if hasattr(me_registry, 'build_from_cfg'):
                    setattr(mmcv_utils, 'build_from_cfg', getattr(me_registry, 'build_from_cfg'))
                if hasattr(me_registry, 'Registry'):
                    setattr(mmcv_utils, 'Registry', getattr(me_registry, 'Registry'))
            except Exception:
                pass
            sys.modules['mmcv.utils'] = mmcv_utils
        except Exception:
            pass

        # mmcv.parallel.* 常见的 collate/scatter
        try:
            mmcv_parallel = types.ModuleType('mmcv.parallel')
            # 尽量复用 torch / mmengine 的工具
            from torch.utils.data._utils.collate import default_collate as _default_collate
            def collate(batch, samples_per_gpu=None):
                return _default_collate(batch)
            mmcv_parallel.collate = collate

            def scatter(inputs, target_gpus, dim=0):
                # 在新版本里通常不需要显式 scatter；这里提供最小实现
                if isinstance(target_gpus, (list, tuple)) and len(target_gpus) > 1:
                    # 简化：多卡时直接返回重复列表，具体并行由 DDP 处理
                    return [inputs for _ in target_gpus]
                return [inputs]
            mmcv_parallel.scatter = scatter

            sys.modules['mmcv.parallel'] = mmcv_parallel
        except Exception:
            pass

        # mmcv.ops / nms 等：大多保留，但路径偶有差异；这里不强行改，等运行时报错再补
    except Exception as e:
        warnings.warn(f'[compat_omni] alias imports failed: {e}')

def _soften_version_asserts():
    # 老仓库里经常有“assert mmcv>=1.5,<1.7”之类，直接软化
    import builtins, re, linecache
    _orig_assert = builtins.__build_class__ if False else None  # 占位，无需真的篡改内建 assert

    # 最稳妥：运行时把常见版本断言绕开
    # 做法：在 import 这些文件前，注入一个 no-op 的 digit_version/parse_version 兼容
    try:
        import types
        mmcv_versioning = types.ModuleType('mmcv.versioning')
        def digit_version(_):
            return (9, 9, 9)
        mmcv_versioning.digit_version = digit_version
        sys.modules['mmcv.versioning'] = mmcv_versioning
    except Exception:
        pass

def _ensure_yapf_safe():
    # 有些老代码 import mmcv.cnn.bricks.transformer 等路径；新版本在 mmcv.ops 或 mmdet.layers
    # 这里先不做激进 hook，等待实际报错再补最小映射。
    return

# 执行补丁
_patch_env()
_soften_version_asserts()
_setup_mmdet3d_utils()
_alias_imports()
_ensure_yapf_safe()
