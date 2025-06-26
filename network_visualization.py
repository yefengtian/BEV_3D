#!/usr/bin/env python3
"""
网络架构可视化脚本
生成BEV 3D感知网络的详细架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_network_architecture_diagram():
    """创建网络架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#E8F4FD',
        'backbone': '#FFE6E6',
        'fpn': '#E6FFE6',
        'transformer': '#FFF2E6',
        'bev_encoder': '#F0E6FF',
        'task_head': '#FFE6F0',
        'output': '#E6F0FF'
    }
    
    # 绘制输入
    input_box = FancyBboxPatch((0.5, 10.5), 9, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 11, '输入: 4个相机图像\n[B, 4, 3, 544, 960]', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 绘制图像编码器
    backbone_box = FancyBboxPatch((0.5, 8.5), 4, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['backbone'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(2.5, 9.25, 'ResNet-50 Backbone\n多尺度特征提取', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    fpn_box = FancyBboxPatch((5.5, 8.5), 4, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['fpn'], 
                            edgecolor='black', linewidth=2)
    ax.add_patch(fpn_box)
    ax.text(7.5, 9.25, 'CustomFPN\n特征融合', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接线
    ax.arrow(5, 10.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(5, 10.5, 0.5, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 绘制视图变换器
    transformer_box = FancyBboxPatch((2, 6.5), 6, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['transformer'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(5, 7.25, 'LSSViewTransformerBEVDepth\n2D→3D→BEV投影', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接线
    ax.arrow(2.5, 8.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.5, 8.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 绘制BEV编码器
    bev_backbone_box = FancyBboxPatch((0.5, 4.5), 4, 1.5, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['bev_encoder'], 
                                     edgecolor='black', linewidth=2)
    ax.add_patch(bev_backbone_box)
    ax.text(2.5, 5.25, 'CustomResNet\nBEV特征提取', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    bev_fpn_box = FancyBboxPatch((5.5, 4.5), 4, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['bev_encoder'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(bev_fpn_box)
    ax.text(7.5, 5.25, 'FPN_LSS\nBEV特征融合', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接线
    ax.arrow(5, 6.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 绘制任务头
    occ_head_box = FancyBboxPatch((0.5, 2.5), 4, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['task_head'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(occ_head_box)
    ax.text(2.5, 3.25, 'BEVOCCHead2D_V2\n占用栅格分割 (13类)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    kps_head_box = FancyBboxPatch((5.5, 2.5), 4, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['task_head'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(kps_head_box)
    ax.text(7.5, 3.25, 'Centerness_Head2D\n停车场关键点检测', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接线
    ax.arrow(2.5, 4.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.5, 4.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 绘制输出
    output_box = FancyBboxPatch((0.5, 0.5), 9, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1, '输出: 占用栅格 + 停车场检测结果', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 连接线
    ax.arrow(2.5, 2.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.5, 2.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 添加标题
    ax.text(5, 11.8, 'BEV 3D感知网络架构图', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加数据流标注
    ax.text(0.2, 9.5, '多视角特征\n[B, 4, 256, H, W]', fontsize=8, rotation=90)
    ax.text(0.2, 7.5, 'BEV特征\n[B, 128, 200, 200]', fontsize=8, rotation=90)
    ax.text(0.2, 5.5, 'BEV特征\n[B, 256, 200, 200]', fontsize=8, rotation=90)
    ax.text(0.2, 3.5, '占用栅格\n[B, 13, H, W]', fontsize=8, rotation=90)
    ax.text(9.8, 3.5, '关键点\n[B, 9, H, W]', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """创建数据流图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 绘制数据流
    stages = [
        {'name': '输入数据', 'pos': (1, 8), 'size': (2, 1), 'color': '#E8F4FD'},
        {'name': '图像预处理', 'pos': (4, 8), 'size': (2, 1), 'color': '#FFE6E6'},
        {'name': '特征提取', 'pos': (7, 8), 'size': (2, 1), 'color': '#E6FFE6'},
        {'name': '视图变换', 'pos': (10, 8), 'size': (2, 1), 'color': '#FFF2E6'},
        {'name': 'BEV编码', 'pos': (7, 6), 'size': (2, 1), 'color': '#F0E6FF'},
        {'name': '占用栅格头', 'pos': (4, 4), 'size': (2, 1), 'color': '#FFE6F0'},
        {'name': '停车场头', 'pos': (10, 4), 'size': (2, 1), 'color': '#FFE6F0'},
        {'name': '后处理', 'pos': (7, 2), 'size': (2, 1), 'color': '#E6F0FF'},
        {'name': '最终输出', 'pos': (7, 0.5), 'size': (2, 1), 'color': '#E6F0FF'}
    ]
    
    for stage in stages:
        box = FancyBboxPatch(stage['pos'], stage['size'][0], stage['size'][1],
                           boxstyle="round,pad=0.1", 
                           facecolor=stage['color'], 
                           edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(stage['pos'][0] + stage['size'][0]/2, 
               stage['pos'][1] + stage['size'][1]/2, 
               stage['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制连接线
    connections = [
        ((3, 8.5), (4, 8.5)),  # 输入 -> 预处理
        ((6, 8.5), (7, 8.5)),  # 预处理 -> 特征提取
        ((9, 8.5), (10, 8.5)), # 特征提取 -> 视图变换
        ((11, 7.5), (8, 7.5)), # 视图变换 -> BEV编码
        ((8, 6.5), (5, 5.5)),  # BEV编码 -> 占用栅格头
        ((8, 6.5), (11, 5.5)), # BEV编码 -> 停车场头
        ((5, 4.5), (8, 3.5)),  # 占用栅格头 -> 后处理
        ((11, 4.5), (8, 3.5)), # 停车场头 -> 后处理
        ((8, 2.5), (8, 1.5))   # 后处理 -> 最终输出
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 添加数据尺寸标注
    data_sizes = [
        ((2, 8.5), '4×[3,544,960]'),
        ((5.5, 8.5), '标准化'),
        ((8.5, 8.5), '4×[256,H,W]'),
        ((11.5, 8.5), '[128,200,200]'),
        ((8.5, 6.5), '[256,200,200]'),
        ((5.5, 4.5), '[13,H,W]'),
        ((11.5, 4.5), '[9,H,W]'),
        ((8.5, 2.5), 'NMS'),
        ((8.5, 0.5), '检测结果')
    ]
    
    for pos, text in data_sizes:
        ax.text(pos[0], pos[1], text, fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.text(6, 9.5, 'BEV 3D感知数据流图', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_function_diagram():
    """创建损失函数图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 绘制损失函数结构
    losses = [
        {'name': '总损失', 'pos': (5, 7), 'size': (2, 0.8), 'color': '#FFE6E6'},
        {'name': '占用栅格损失', 'pos': (2, 5.5), 'size': (2, 0.8), 'color': '#E6FFE6'},
        {'name': '停车场检测损失', 'pos': (6, 5.5), 'size': (2, 0.8), 'color': '#E6FFE6'},
        {'name': '深度损失', 'pos': (4, 4), 'size': (2, 0.8), 'color': '#FFF2E6'},
        {'name': '分类损失', 'pos': (6, 3), 'size': (1.5, 0.8), 'color': '#F0E6FF'},
        {'name': '回归损失', 'pos': (8, 3), 'size': (1.5, 0.8), 'color': '#F0E6FF'},
        {'name': 'Focal Loss', 'pos': (2, 2), 'size': (1.5, 0.8), 'color': '#FFE6F0'},
        {'name': 'Gaussian Focal', 'pos': (6, 1.5), 'size': (1.5, 0.8), 'color': '#FFE6F0'},
        {'name': 'L1 Loss', 'pos': (8, 1.5), 'size': (1.5, 0.8), 'color': '#FFE6F0'}
    ]
    
    for loss in losses:
        box = FancyBboxPatch(loss['pos'], loss['size'][0], loss['size'][1],
                           boxstyle="round,pad=0.1", 
                           facecolor=loss['color'], 
                           edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(loss['pos'][0] + loss['size'][0]/2, 
               loss['pos'][1] + loss['size'][1]/2, 
               loss['name'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 绘制连接线
    connections = [
        ((6, 7.4), (3, 6.3)),  # 总损失 -> 占用栅格损失
        ((6, 7.4), (7, 6.3)),  # 总损失 -> 停车场检测损失
        ((3, 5.5), (5, 4.8)),  # 占用栅格损失 -> 深度损失
        ((7, 5.5), (6.75, 4.8)), # 停车场检测损失 -> 分类损失
        ((7, 5.5), (8.75, 4.8)), # 停车场检测损失 -> 回归损失
        ((3, 5.5), (2.75, 3.2)), # 占用栅格损失 -> Focal Loss
        ((6.75, 3.8), (6.75, 2.7)), # 分类损失 -> Gaussian Focal
        ((8.75, 3.8), (8.75, 2.7))  # 回归损失 -> L1 Loss
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # 添加权重标注
    weights = [
        ((3.5, 6.5), '权重: 1.0'),
        ((7.5, 6.5), '权重: 1.0'),
        ((5.5, 4.5), '权重: 1.0'),
        ((2.5, 2.5), '权重: 1.0'),
        ((6.5, 1.5), '权重: 1.0'),
        ((8.5, 1.5), '权重: 0.25')
    ]
    
    for pos, text in weights:
        ax.text(pos[0], pos[1], text, fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.7))
    
    ax.text(5, 7.8, '损失函数结构图', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('loss_function_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """生成所有网络架构图"""
    print("生成网络架构图...")
    create_network_architecture_diagram()
    
    print("生成数据流图...")
    create_data_flow_diagram()
    
    print("生成损失函数图...")
    create_loss_function_diagram()
    
    print("所有图表已生成完成！")
    print("文件列表:")
    print("- network_architecture.png: 网络架构图")
    print("- data_flow_diagram.png: 数据流图")
    print("- loss_function_diagram.png: 损失函数图")

if __name__ == '__main__':
    main() 