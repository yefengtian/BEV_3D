import numpy as np
import os

def try_read_bin_file(filepath):
    assert os.path.isfile(filepath), f"[ERROR] 文件不存在: {filepath}"
    
    with open(filepath, 'rb') as f:
        byte_data = f.read()
    
    file_size = len(byte_data)
    print(f"[INFO] 文件大小: {file_size} 字节")

    valid_field_counts = []
    for num_fields in range(3, 9):  # float32个数
        if file_size % (4 * num_fields) == 0:
            valid_field_counts.append(num_fields)

    if not valid_field_counts:
        print("[ERROR] 无法匹配任何可能字段数，请检查文件是否损坏或格式不一致")
        return

    print(f"[INFO] 可能的字段数（float32）: {valid_field_counts}")
    
    # 默认选用最小字段数来试读（用户可更换）
    field_count = valid_field_counts[0]
    print(f"[INFO] 默认尝试读取字段数: {field_count}（你可手动修改）")

    data = np.frombuffer(byte_data, dtype=np.float32).reshape(-1, field_count)
    print(f"[INFO] 实际点数: {data.shape[0]}, 每个点包含字段数: {data.shape[1]}")

    # 打印前几个点
    print("[DEBUG] 前5个点示例:")
    for i in range(min(5, data.shape[0])):
        fields = ", ".join([f"{v:.3f}" for v in data[i]])
        print(f"  点 {i}: [{fields}]")

    # 分析 xyz
    if field_count >= 3:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        print("[INFO] 点云范围:")
        print(f"  X: min={x.min():.2f}, max={x.max():.2f}")
        print(f"  Y: min={y.min():.2f}, max={y.max():.2f}")
        print(f"  Z: min={z.min():.2f}, max={z.max():.2f}")
    if field_count >= 4:
        reflectance = data[:, 3]
        print(f"  Reflectance: min={reflectance.min():.2f}, max={reflectance.max():.2f}")
    if field_count > 4:
        print(f"  Extra fields [{field_count - 4}]:")
        for i in range(4, field_count):
            print(f"    Field {i}: min={data[:, i].min():.2f}, max={data[:, i].max():.2f}")

    return data

if __name__ == "__main__":
    test_path = "your_lidar_file.bin"  # 修改为实际路径
    try_read_bin_file(test_path)
