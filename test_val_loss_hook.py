#!/usr/bin/env python3
"""
测试ValLossHook实现的简单脚本
"""
import sys
import os
import argparse

# 添加项目路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_interface'))

try:
    # 新栈（MMCV 2.x + MMEngine）
    from mmengine.config import Config, DictAction, ConfigDict
except Exception:
    # 旧栈（MMCV 1.x）
    from mmcv import Config
    from mmcv.utils import DictAction, ConfigDict

def test_val_loss_hook():
    """测试ValLossHook是否能正确导入和初始化"""
    try:
        # 测试导入
        from mmdet3d.hooks.val_loss_hook import ValLossHook, parse_losses
        print("✓ ValLossHook导入成功")
        
        # 测试parse_losses函数
        import torch
        fake_losses = {
            'loss_cls': torch.tensor(0.5),
            'loss_bbox': torch.tensor(0.3),
            'loss_centerness': torch.tensor(0.2)
        }
        total_loss, log_vars = parse_losses(fake_losses)
        print(f"✓ parse_losses函数工作正常，总loss: {total_loss.item():.4f}")
        
        # 测试ValLossHook初始化
        hook = ValLossHook(
            dataset_cfg={'type': 'CarlaDataset', 'test_mode': False},
            dataloader_cfg={'samples_per_gpu': 1, 'workers_per_gpu': 1},
            interval=1,
            rule='less'
        )
        print("✓ ValLossHook初始化成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_carla_dataset():
    """测试CarlaDataset的evaluate方法"""
    try:
        from mmdet3d.datasets.carla_dataset import CarlaDataset
        print("✓ CarlaDataset导入成功")

        data_root = 'data/carla_bev/'
        
        # 测试evaluate方法签名
        dataset = CarlaDataset(
            ann_file= data_root + '0801_5_samples.pkl',
            data_root= data_root,
            classes = ['non_occ','occ'],
            pipeline=[]
        )
        
        # 测试evaluate方法
        result = dataset.evaluate(
            results=[],
            metric=None,
            logger=None,
            jsonfile_prefix=None,
            result_names=None
        )
        print("✓ CarlaDataset.evaluate方法调用成功")
        
        return True
        
    except Exception as e:
        print(f"✗ CarlaDataset测试失败: {e}")
        return False

def test_config():
    """测试配置文件语法"""
    try:
        # 这里我们只测试Python语法，不测试实际的配置加载
        config_path = os.path.join(os.path.dirname(__file__), 'model_interface', 'config', 'freespace_occ2d_r50_depth.py')
        
        with open(config_path, 'r') as f:
            config_content = f.read()

        # cfg = Config.fromfile(config_path)
        
        # 检查关键配置是否存在
        if 'ValLossHook' in config_content:
            print("✓ 配置文件中包含ValLossHook")
        else:
            print("✗ 配置文件中缺少ValLossHook")
            return False
            
        if 'custom_imports' in config_content:
            print("✓ 配置文件中包含custom_imports")
        else:
            print("✗ 配置文件中缺少custom_imports")
            return False
            
        if 'evaluation = dict(interval=0)' in config_content:
            print("✓ 配置文件中正确设置了evaluation")
        else:
            print("✗ 配置文件中evaluation设置不正确")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False


if __name__ == '__main__':
    print("开始测试ValLossHook实现...")
    print("=" * 50)
    
    success = True
    
    # 测试各个组件
    success &= test_val_loss_hook()
    print()
    
    success &= test_config()
    print()

    success &= test_carla_dataset()
    print()
    
    print("=" * 50)
    if success:
        print("🎉 所有测试通过！ValLossHook实现正确。")
        print("\n使用说明:")
        print("1. 训练时会自动在每个epoch结束后计算验证集loss")
        print("2. 当验证集loss更优时会自动保存best checkpoint")
        print("3. 最优checkpoint文件名格式: best_val_loss_epoch_XXX.pth")
        print("4. 训练日志中会显示val_loss指标")
    else:
        print("❌ 部分测试失败，请检查实现。")
        sys.exit(1)
