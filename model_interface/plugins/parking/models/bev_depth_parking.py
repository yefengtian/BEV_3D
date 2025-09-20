import torch
import torch.nn.functional as F

# --- Compat shim: mmcv1 decorators -> no-op on mmcv2 ---
def _noop_decorator(*args, **kwargs):
    def wrapper(fn):
        return fn
    return wrapper

try:
    # 如果真是 mmcv1 环境，会走这里；mmcv2 会抛异常然后走 no-op
    from mmcv.runner import auto_fp16 as _auto_fp16, force_fp32 as _force_fp32  # type: ignore
    auto_fp16 = _auto_fp16
    force_fp32 = _force_fp32
except Exception:
    auto_fp16 = _noop_decorator
    force_fp32 = _noop_decorator
# --- end shim ---


# 使用官方 mmdet3d 的 registry
from mmdet3d.registry import MODELS

import numpy as np


@MODELS.register_module()
class BEVDepthParking:
    """BEVDepthParking 模型，组合包装器版本"""
    
    def __init__(self,
                 inner_model=None,
                 data_preprocessor=None,
                 **kwargs):
        """
        Args:
            inner_model: 内部模型配置，可以是 dict 或已构建的模型
            data_preprocessor: 数据预处理器
            **kwargs: 其他参数（为了兼容原配置）
        """
        if inner_model is None:
            # 如果没有提供 inner_model，尝试从 kwargs 构建一个简化的 BEVDepth
            inner_model = self._build_default_model(kwargs)
        
        if isinstance(inner_model, (dict, type(None))):
            self.inner = MODELS.build(inner_model) if inner_model else None
        else:
            self.inner = inner_model
            
        self.data_preprocessor = data_preprocessor
        
        # 为了兼容性，保留一些属性
        self.upsample = kwargs.get('upsample', False)
        self.with_img_neck = True
        self.with_pts_bbox = False
        
    def _build_default_model(self, kwargs):
        """构建默认的内部模型"""
        # 这里可以构建一个简化的 BEVDepth 模型
        # 暂时返回 None，让用户通过 inner_model 参数传入
        return None

    def forward(self, *args, **kwargs):
        """转发到内部模型"""
        if self.inner is not None:
            return self.inner(*args, **kwargs)
        else:
            # 如果没有内部模型，返回一个简单的占位符
            return {'loss': torch.tensor(0.0, requires_grad=True)}

    def forward_train(self, *args, **kwargs):
        """训练前向"""
        return self.forward(*args, **kwargs)

    def forward_test(self, *args, **kwargs):
        """测试前向"""
        return self.forward(*args, **kwargs)

    def simple_test(self, *args, **kwargs):
        """简单测试"""
        return self.forward(*args, **kwargs)

    def forward_dummy(self, *args, **kwargs):
        """虚拟前向"""
        return self.forward(*args, **kwargs)
