#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的代码是否能正常运行
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """测试导入是否成功"""
    print("=== 测试导入 ===")
    
    try:
        # 测试官方 mmdet3d 导入
        from mmdet3d.utils import register_all_modules
        register_all_modules()
        print("✓ 官方 mmdet3d 注册成功")
    except Exception as e:
        print(f"✗ 官方 mmdet3d 注册失败: {e}")
        return False
    
    try:
        # 测试插件导入
        from model_interface.plugins.parking import models, datasets
        print("✓ 插件导入成功")
    except Exception as e:
        print(f"✗ 插件导入失败: {e}")
        return False
    
    return True

def test_registration():
    """测试注册是否成功"""
    print("\n=== 测试注册 ===")
    
    try:
        from mmdet3d.registry import MODELS, DATASETS
        
        # 检查 BEVDepthParking 是否已注册
        if 'BEVDepthParking' in MODELS._module_dict:
            print("✓ BEVDepthParking 已注册")
        else:
            print("✗ BEVDepthParking 未注册")
            return False
        
        # 检查 CarlaDataset 是否已注册
        if 'CarlaDataset' in DATASETS._module_dict:
            print("✓ CarlaDataset 已注册")
        else:
            print("✗ CarlaDataset 未注册")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 注册检查失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件加载 ===")
    
    try:
        from mmengine.config import Config
        
        config_path = 'model_interface/config/freespace_occ2d_r50_depth.py'
        cfg = Config.fromfile(config_path)
        
        print(f"✓ 配置文件加载成功: {config_path}")
        print(f"  - 模型类型: {cfg.model.type}")
        print(f"  - 数据集类型: {cfg.dataset_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_model_building():
    """测试模型构建"""
    print("\n=== 测试模型构建 ===")
    
    try:
        from mmdet3d.registry import MODELS
        from mmengine.config import Config
        
        config_path = 'model_interface/config/freespace_occ2d_r50_depth.py'
        cfg = Config.fromfile(config_path)
        
        # 尝试构建模型
        model = MODELS.build(cfg.model)
        print("✓ 模型构建成功")
        print(f"  - 模型类型: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        return False

def main():
    """主函数"""
    print("开始测试修复后的代码...")
    
    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败")
        return False
    
    # 测试注册
    if not test_registration():
        print("\n❌ 注册测试失败")
        return False
    
    # 测试配置文件加载
    if not test_config_loading():
        print("\n❌ 配置文件加载失败")
        return False
    
    # 测试模型构建
    if not test_model_building():
        print("\n❌ 模型构建失败")
        return False
    
    print("\n✅ 所有测试通过！修复成功")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
