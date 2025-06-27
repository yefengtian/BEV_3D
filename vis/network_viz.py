#!/usr/bin/env python3
"""
简化的网络架构可视化脚本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_simple_architecture():
    """创建简化的网络架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 定义组件
    components = [
        {'name': '输入\n4个相机图像', 'pos': (1, 6), 'size': (2, 1)},
        {'name': 'ResNet-50\nBackbone', 'pos': (4, 6), 'size': (2, 1)},
        {'name': 'CustomFPN\n特征融合', 'pos': (7, 6), 'size': (2, 1)},
        {'name': 'LSS视图变换器\n2D→3D→BEV', 'pos': (5, 4), 'size': (2, 1)},
        {'name': 'BEV编码器\nCustomResNet+FPN', 'pos': (5, 2), 'size': (2, 1)},
        {'name': '占用栅格头\n13类分割', 'pos': (2, 0.5), 'size': (2, 1)},
        {'name': '停车场头\n关键点检测', 'pos': (7, 0.5), 'size': (2, 1)}
    ]
    
    # 绘制组件
    for comp in components:
        rect = patches.Rectangle(comp['pos'], comp['size'][0], comp['size'][1], 
                               linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2,
               comp['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制连接线
    connections = [
        ((3, 6.5), (4, 6.5)),  # 输入 -> Backbone
        ((6, 6.5), (7, 6.5)),  # Backbone -> FPN
        ((8, 6.5), (6, 5.5)),  # FPN -> 视图变换器
        ((6, 4.5), (6, 3.5)),  # 视图变换器 -> BEV编码器
        ((6, 2.5), (3, 1.5)),  # BEV编码器 -> 占用栅格头
        ((6, 2.5), (8, 1.5))   # BEV编码器 -> 停车场头
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.text(5, 7.5, 'BEV 3D感知网络架构', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('simple_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_simple_architecture() 