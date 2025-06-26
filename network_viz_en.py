#!/usr/bin/env python3
"""
English version of network architecture visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    """Create network architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define components
    components = [
        {'name': 'Input\n4 Camera Images', 'pos': (1, 8), 'size': (2, 1), 'color': '#E8F4FD'},
        {'name': 'ResNet-50\nBackbone', 'pos': (4, 8), 'size': (2, 1), 'color': '#FFE6E6'},
        {'name': 'CustomFPN\nFeature Fusion', 'pos': (7, 8), 'size': (2, 1), 'color': '#E6FFE6'},
        {'name': 'LSS View\nTransformer', 'pos': (10, 8), 'size': (2, 1), 'color': '#FFF2E6'},
        {'name': 'BEV Encoder\nCustomResNet+FPN', 'pos': (7, 6), 'size': (2, 1), 'color': '#F0E6FF'},
        {'name': 'Occupancy Head\n13-class Segmentation', 'pos': (4, 4), 'size': (2, 1), 'color': '#FFE6F0'},
        {'name': 'Parking Head\nKeypoint Detection', 'pos': (10, 4), 'size': (2, 1), 'color': '#FFE6F0'},
        {'name': 'Post-processing\nNMS', 'pos': (7, 2), 'size': (2, 1), 'color': '#E6F0FF'},
        {'name': 'Final Output\nDetection Results', 'pos': (7, 0.5), 'size': (2, 1), 'color': '#E6F0FF'}
    ]
    
    # Draw components
    for comp in components:
        rect = patches.Rectangle(comp['pos'], comp['size'][0], comp['size'][1], 
                               linewidth=2, edgecolor='black', facecolor=comp['color'])
        ax.add_patch(rect)
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2,
               comp['name'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connections
    connections = [
        ((3, 8.5), (4, 8.5)),  # Input -> Backbone
        ((6, 8.5), (7, 8.5)),  # Backbone -> FPN
        ((9, 8.5), (10, 8.5)), # FPN -> View Transformer
        ((11, 7.5), (8, 7.5)), # View Transformer -> BEV Encoder
        ((8, 6.5), (5, 5.5)),  # BEV Encoder -> Occupancy Head
        ((8, 6.5), (11, 5.5)), # BEV Encoder -> Parking Head
        ((5, 4.5), (8, 3.5)),  # Occupancy Head -> Post-processing
        ((11, 4.5), (8, 3.5)), # Parking Head -> Post-processing
        ((8, 2.5), (8, 1.5))   # Post-processing -> Final Output
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add data flow annotations
    data_flows = [
        ((2, 8.5), '[B,4,3,544,960]'),
        ((5.5, 8.5), '[B,4,256,H,W]'),
        ((8.5, 8.5), '[B,4,256,H,W]'),
        ((11.5, 8.5), '[B,128,200,200]'),
        ((8.5, 6.5), '[B,256,200,200]'),
        ((5.5, 4.5), '[B,13,H,W]'),
        ((11.5, 4.5), '[B,9,H,W]'),
        ((8.5, 2.5), 'NMS'),
        ((8.5, 0.5), 'Results')
    ]
    
    for pos, text in data_flows:
        ax.text(pos[0], pos[1], text, fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.text(6, 9.5, 'BEV 3D Perception Network Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('network_architecture_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """Create data flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define stages
    stages = [
        {'name': 'Input Data', 'pos': (1, 6), 'size': (2, 1), 'color': '#E8F4FD'},
        {'name': 'Image Preprocessing', 'pos': (4, 6), 'size': (2, 1), 'color': '#FFE6E6'},
        {'name': 'Feature Extraction', 'pos': (7, 6), 'size': (2, 1), 'color': '#E6FFE6'},
        {'name': 'View Transformation', 'pos': (5, 4), 'size': (2, 1), 'color': '#FFF2E6'},
        {'name': 'BEV Encoding', 'pos': (5, 2), 'size': (2, 1), 'color': '#F0E6FF'},
        {'name': 'Task Heads', 'pos': (5, 0.5), 'size': (2, 1), 'color': '#FFE6F0'}
    ]
    
    # Draw stages
    for stage in stages:
        rect = patches.Rectangle(stage['pos'], stage['size'][0], stage['size'][1], 
                               linewidth=2, edgecolor='black', facecolor=stage['color'])
        ax.add_patch(rect)
        ax.text(stage['pos'][0] + stage['size'][0]/2, stage['pos'][1] + stage['size'][1]/2,
               stage['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw connections
    connections = [
        ((3, 6.5), (4, 6.5)),  # Input -> Preprocessing
        ((6, 6.5), (7, 6.5)),  # Preprocessing -> Feature Extraction
        ((8, 6.5), (6, 5.5)),  # Feature Extraction -> View Transformation
        ((6, 4.5), (6, 3.5)),  # View Transformation -> BEV Encoding
        ((6, 2.5), (6, 1.5))   # BEV Encoding -> Task Heads
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add data dimensions
    dimensions = [
        ((2, 6.5), '4×[3,544,960]'),
        ((5.5, 6.5), 'Normalized'),
        ((8.5, 6.5), '4×[256,H,W]'),
        ((6.5, 4.5), '[128,200,200]'),
        ((6.5, 2.5), '[256,200,200]'),
        ((6.5, 0.5), '[13+9,H,W]')
    ]
    
    for pos, text in dimensions:
        ax.text(pos[0], pos[1], text, fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.text(5, 7.5, 'Data Flow Diagram', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_flow_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_structure():
    """Create loss function structure diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define loss components
    losses = [
        {'name': 'Total Loss', 'pos': (5, 7), 'size': (2, 0.8), 'color': '#FFE6E6'},
        {'name': 'Occupancy Loss', 'pos': (2, 5.5), 'size': (2, 0.8), 'color': '#E6FFE6'},
        {'name': 'Parking Loss', 'pos': (6, 5.5), 'size': (2, 0.8), 'color': '#E6FFE6'},
        {'name': 'Depth Loss', 'pos': (4, 4), 'size': (2, 0.8), 'color': '#FFF2E6'},
        {'name': 'Classification\nLoss', 'pos': (6, 3), 'size': (1.5, 0.8), 'color': '#F0E6FF'},
        {'name': 'Regression\nLoss', 'pos': (8, 3), 'size': (1.5, 0.8), 'color': '#F0E6FF'},
        {'name': 'Focal Loss', 'pos': (2, 2), 'size': (1.5, 0.8), 'color': '#FFE6F0'},
        {'name': 'Gaussian\nFocal Loss', 'pos': (6, 1.5), 'size': (1.5, 0.8), 'color': '#FFE6F0'},
        {'name': 'L1 Loss', 'pos': (8, 1.5), 'size': (1.5, 0.8), 'color': '#FFE6F0'}
    ]
    
    # Draw loss components
    for loss in losses:
        rect = patches.Rectangle(loss['pos'], loss['size'][0], loss['size'][1], 
                               linewidth=2, edgecolor='black', facecolor=loss['color'])
        ax.add_patch(rect)
        ax.text(loss['pos'][0] + loss['size'][0]/2, loss['pos'][1] + loss['size'][1]/2,
               loss['name'], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw connections
    connections = [
        ((6, 7.4), (3, 6.3)),  # Total -> Occupancy
        ((6, 7.4), (7, 6.3)),  # Total -> Parking
        ((3, 5.5), (5, 4.8)),  # Occupancy -> Depth
        ((7, 5.5), (6.75, 4.8)), # Parking -> Classification
        ((7, 5.5), (8.75, 4.8)), # Parking -> Regression
        ((3, 5.5), (2.75, 3.2)), # Occupancy -> Focal
        ((6.75, 3.8), (6.75, 2.7)), # Classification -> Gaussian Focal
        ((8.75, 3.8), (8.75, 2.7))  # Regression -> L1
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # Add weights
    weights = [
        ((3.5, 6.5), 'Weight: 1.0'),
        ((7.5, 6.5), 'Weight: 1.0'),
        ((5.5, 4.5), 'Weight: 1.0'),
        ((2.5, 2.5), 'Weight: 1.0'),
        ((6.5, 1.5), 'Weight: 1.0'),
        ((8.5, 1.5), 'Weight: 0.25')
    ]
    
    for pos, text in weights:
        ax.text(pos[0], pos[1], text, fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.7))
    
    ax.text(5, 7.8, 'Loss Function Structure', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('loss_structure_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all diagrams"""
    print("Generating network architecture diagram...")
    create_architecture_diagram()
    
    print("Generating data flow diagram...")
    create_data_flow_diagram()
    
    print("Generating loss structure diagram...")
    create_loss_structure()
    
    print("All diagrams generated successfully!")
    print("Files created:")
    print("- network_architecture_en.png: Network architecture diagram")
    print("- data_flow_en.png: Data flow diagram")
    print("- loss_structure_en.png: Loss function structure diagram")

if __name__ == '__main__':
    main() 