import numpy as np
import matplotlib.pyplot as plt

# 1) 合成 LiDAR 点云：5000 个点，x∈[-10,10], y∈[-2,2], z∈[1,50]
num_points = 5000
pts_lidar = np.vstack((
    np.random.uniform(-10, 10, num_points),
    np.random.uniform(-2, 2, num_points),
    np.random.uniform(1, 50, num_points)
)).T

# 2) 简单相机内参
fx, fy = 800, 800
cx, cy = 352, 128
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# 3) 图像尺寸与下采样
H, W = 256, 704
downsample = 4
Hf, Wf = H // downsample, W // downsample

# 4) 投影与深度图生成
pts_cam = pts_lidar  # 假设已在相机坐标系
uv = (K @ pts_cam.T).T
u, v, z = uv[:,0]/uv[:,2], uv[:,1]/uv[:,2], pts_cam[:,2]
depth_map = np.full((Hf, Wf), np.nan)
for ui, vi, zi in zip(u, v, z):
    if 0 <= ui < W and 0 <= vi < H:
        ui_ds, vi_ds = int(ui/downsample), int(vi/downsample)
        prev = depth_map[vi_ds, ui_ds]
        depth_map[vi_ds, ui_ds] = zi if np.isnan(prev) else min(prev, zi)

# 5) 可视化
plt.imshow(depth_map)
plt.title("Sparse Depth Map from LiDAR Projection")
plt.xlabel("Wf")
plt.ylabel("Hf")
plt.colorbar(label="Depth (m)")
plt.show()
