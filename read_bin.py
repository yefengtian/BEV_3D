import numpy as np
import struct
import os

def read_lidar_bin_file(filepath, debug=True):
    assert os.path.isfile(filepath), f"[ERROR] 文件不存在: {filepath}"

    # 读取原始二进制数据
    with open(filepath, 'rb') as f:
        byte_data = f.read()

    point_byte_size = 4 * 4  # x, y, z, reflectance, 每个为 float32 = 4 bytes
    total_points = len(byte_data) // point_byte_size

    if debug:
        print(f"[INFO] 打开文件: {filepath}")
        print(f"[INFO] 文件大小: {len(byte_data)} 字节")
        print(f"[INFO] 每个点占用 {point_byte_size} 字节，理论点数: {total_points}")

    # 将字节流转换为 NumPy array
    point_data = np.frombuffer(byte_data, dtype=np.float32).reshape(-1, 4)

    if debug:
        print(f"[DEBUG] 实际读取到点的数量: {point_data.shape[0]}")
        print(f"[DEBUG] 每个点包含字段: x, y, z, reflectance")
        print("[DEBUG] 前5个点示例:")
        for i in range(min(5, point_data.shape[0])):
            x, y, z, r = point_data[i]
            print(f"  点 {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}, reflectance={r:.3f}")

        # 输出统计信息
        xyz = point_data[:, :3]
        print("[DEBUG] 点云范围 (x/y/z):")
        print(f"    X: min={xyz[:,0].min():.2f}, max={xyz[:,0].max():.2f}")
        print(f"    Y: min={xyz[:,1].min():.2f}, max={xyz[:,1].max():.2f}")
        print(f"    Z: min={xyz[:,2].min():.2f}, max={xyz[:,2].max():.2f}")
        print(f"[DEBUG] 反射率范围: min={point_data[:,3].min():.2f}, max={point_data[:,3].max():.2f}")

    return point_data

if __name__ == "__main__":
    # 修改成你自己的 bin 文件路径
    test_bin_path = "example/000000.bin"
    points = read_lidar_bin_file(test_bin_path, debug=True)
