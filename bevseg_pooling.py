import cv2
import time
import numpy as np
from PIL import Image
import os
from multiprocessing import Pool, cpu_count

CityScapesPalette = {
    (0, 0, 0): 0,             # Unlabeled
    (128, 64, 128): 1,        # Roads
    (244, 35, 232): 2,        # SideWalks
    (70, 70, 70): 3,          # Building
    (102, 102, 156): 4,       # Wall
    (190, 153, 153): 5,       # Fence
    (153, 153, 153): 6,       # Pole
    (250, 170, 30): 7,        # TrafficLight
    (220, 220, 0): 8,         # TrafficSign
    (107, 142, 35): 9,        # Vegetation
    (152, 251, 152): 10,      # Terrain
    (70, 130, 180): 11,       # Sky
    (220, 20, 60): 12,        # Pedestrian
    (255, 0, 0): 13,          # Rider
    (0, 0, 142): 14,          # Car
    (0, 0, 70): 15,           # Truck
    (0, 60, 100): 16,         # Bus
    (0, 80, 100): 17,         # Train
    (0, 0, 230): 18,          # Motorcycle
    (119, 11, 32): 19,        # Bicycle
    (110, 190, 160): 20,      # Static
    (170, 120, 50): 21,       # Dynamic
    (55, 90, 80): 22,         # Other
    (45, 60, 150): 23,        # Water
    (157, 234, 50): 24,       # RoadLine
    (81, 0, 81): 25,          # Ground
    (150, 100, 100): 26,      # Bridge
    (230, 150, 140): 27,      # RailTrack
    (180, 165, 180): 28,      # GuardRail
    (255, 99, 71): 29,        # Cone
}

CityScapesPaletteDecode = {
    0: (0, 0, 0),             # Unlabeled
    1: (128, 64, 128),        # Roads
    2: (244, 35, 232),        # SideWalks
    3: (70, 70, 70),          # Building
    4: (102, 102, 156),       # Wall
    5: (190, 153, 153),       # Fence
    6: (153, 153, 153),       # Pole
    7: (250, 170, 30),        # TrafficLight
    8: (220, 220, 0),         # TrafficSign
    9: (107, 142, 35),        # Vegetation
    10: (152, 251, 152),      # Terrain
    11: (70, 130, 180),       # Sky
    12: (220, 20, 60),        # Pedestrian
    13: (255, 0, 0),          # Rider
    14: (0, 0, 142),          # Car
    15: (0, 0, 70),           # Truck
    16: (0, 60, 100),         # Bus
    17: (0, 80, 100),         # Train
    18: (0, 0, 230),          # Motorcycle
    19: (119, 11, 32),        # Bicycle
    20: (110, 190, 160),      # Static
    21: (170, 120, 50),       # Dynamic
    22: (55, 90, 80),         # Other
    23: (45, 60, 150),        # Water
    24: (157, 234, 50),       # RoadLine
    25: (81, 0, 81),          # Ground
    26: (150, 100, 100),      # Bridge
    27: (230, 150, 140),      # RailTrack
    28: (180, 165, 180),      # GuardRail
    29: (255, 99, 71),        # Cone
}

FreespacePalette = {
    0: (0, 0, 0),                # unlabeled, black
    1: (169, 169, 169),          # freespace, darkgray
    2: (0, 255, 255),            # sidewalks, aqua
    3: (100, 149, 237),          # building, cornflowerblue
    4: (255, 192, 203),          # fence, pink
    5: (255, 255, 0),            # pole, yellow
    6: (189, 183, 107),          # terrain, darkkhaki
    7: (255, 0, 255),            # pedestrian, fuscia
    8: (123, 104, 238),          # rider, mediumslateblue
    9: (0, 255, 0),              # vehicle, lime
    10: (0, 128, 0),             # train, green
    11: (160, 82, 45),           # others, sienna
    12: (255, 250, 250),         # roadline, snow
    13: (255, 69, 0),            # cone, orangered
}

FreespaceIDMapping = {
    0: 0,   # unlabeled -> unlabeled
    1: 1,   # roads -> freespace
    2: 1,   # sidewalks -> freespace         The parking lot and the road are connected via sidewalks only
    3: 3,   # building -> building
    4: 3,   # wall -> building
    5: 4,   # fence -> fence
    6: 5,   # pole -> pole
    7: 5,   # trafficlight -> pole
    8: 5,   # trafficsign -> pole
    9: 6,   # vegetation -> terrain
    10: 1,  # terrain -> freespace           Road surface of parking lots are marked as terrain in CarlaTown
    11: 0,  # sky -> unlabeled
    12: 7,  # pedestrian -> pedestrian
    13: 8,  # rider -> rider
    14: 9,  # car -> vehicle
    15: 9,  # truck -> vehicle
    16: 9,  # bus -> vehicle
    17: 10, # train -> train
    18: 8,  # motorcycle -> rider
    19: 8,  # bicycle -> rider
    20: 11, # static -> others
    21: 11, # dynamic -> others
    22: 11, # other -> others
    23: 6,  # water -> terrain
    24: 12, # roadline -> roadline          Parking slots borders are marked as roadline in CarlaTown. So preserved.
    25: 1,  # ground -> freespace
    26: 3,  # bridge -> building
    27: 3,  # railtrack -> building
    28: 4,  # guardrail -> fence
    29: 13, # cone -> cone
}

def semantic_pooling(semantic_map, factor=10, obstacle_labels=None, drivable_labels=None):
    """
    Downsample a semantic map using custom pooling rules.

    Parameters:
    - semantic_map: numpy.ndarray of shape (H, W), integer labels.
    - factor: int, factor by which to downscale the map (default 10).
    - obstacle_labels: list or array of ints, labels considered as obstacles.
    - drivable_labels: list or array of ints, labels considered as drivable.

    Returns:
    - downsampled_map: numpy.ndarray of shape (H//factor, W//factor), pooled semantic labels.
    """
    # 检查输入参数
    assert obstacle_labels is not None and drivable_labels is not None, \
                                                        "Both obstacle_labels and drivable_labels must be provided."
    assert isinstance(semantic_map, np.ndarray) and len(semantic_map.shape) == 2, \
                                                        "semantic_map must be a 2D numpy ndarray."
    h, w = semantic_map.shape
    assert h % factor == 0 and w % factor == 0, "Height and width must be divisible by factor"
    
    # 标签检查
    labels_in_map = np.unique(semantic_map)
    obstacle_labels = np.array(obstacle_labels)
    drivable_labels = np.array(drivable_labels)
    all_specified_labels = np.concatenate((obstacle_labels, drivable_labels))

    if np.intersect1d(obstacle_labels, drivable_labels).size != 0:
        overlapping_labels = np.intersect1d(obstacle_labels, drivable_labels)
        raise ValueError(f"Obstacle labels and drivable labels should not overlap. Overlapping labels: {overlapping_labels}")
   
    if not np.all(np.isin(labels_in_map, all_specified_labels)):
        missing_labels = labels_in_map[~np.isin(labels_in_map, all_specified_labels)]
        raise ValueError(f"Labels not specified in obstacle or drivable labels: {missing_labels}")

    tiles = semantic_map.reshape(h // factor, factor, -1, factor).swapaxes(1, 2).reshape(-1, factor * factor)
    n_tiles = tiles.shape[0]
    n_labels = np.max(all_specified_labels) + 1

    # 统计类别出现次数
    counts = np.zeros((n_tiles, n_labels), dtype=np.int32)
    flat_tiles = tiles.flatten()                                    # 类别索引
    tile_indices = np.repeat(np.arange(n_tiles), factor * factor)   # tile 索引
    combined_indices = tile_indices * n_labels + flat_tiles
    counts_flat = np.bincount(combined_indices, minlength=n_tiles * n_labels)
    counts = counts_flat.reshape(n_tiles, n_labels)
    
    # 统计障碍物和可通行区域的像素计数
    obstacle_counts = counts[:, obstacle_labels]    # (n_tiles, n_obstacle_labels)
    drivable_counts = counts[:, drivable_labels]    # (n_tiles, n_drivable_labels)

    obstacle_indices_with_max_counts = obstacle_counts.argmax(axis=1)
    max_obstacle_labels = obstacle_labels[obstacle_indices_with_max_counts]

    drivable_indices_with_max_counts = drivable_counts.argmax(axis=1)
    max_drivable_labels = drivable_labels[drivable_indices_with_max_counts]

    # 判断障碍物是否存在（像素数超过阈值）
    threshold = int(0.1 * factor * factor)
    total_obstacle_present = obstacle_counts.sum(axis=1) >= threshold  # (n_tiles,)

    # 池化操作
    pooled_labels = np.where(
        total_obstacle_present,
        max_obstacle_labels,
        max_drivable_labels
    )
    downsampled_map = pooled_labels.reshape(h // factor, w // factor)

    return downsampled_map

def process_image(args):
    """
    处理单个图像的函数，用于多进程并行处理。

    参数：
    - args: 包含必要参数的元组 (file_name, folder_path, output_path, obstacle_labels, drivable_labels)
    """
    file_name, folder_path, output_path, obstacle_labels, drivable_labels = args
    if file_name.endswith('.png'):
        file_path = os.path.join(folder_path, file_name)
        demo_semantic_rgb = np.array(Image.open(file_path))
        demo_semantic_rgb = demo_semantic_rgb[:, :, :3]         # 去除 alpha 通道（如果有）
        
        demo_semantic = np.zeros(demo_semantic_rgb.shape[:2], dtype=np.int32)
        # 将 RGB 颜色映射到标签
        for color, label in CityScapesPalette.items():
            mask = np.all(demo_semantic_rgb == color, axis=-1)
            demo_semantic[mask] = label
        
        tic = time.time()
        downsampled_map = semantic_pooling(demo_semantic, 
                                           factor=10, 
                                           obstacle_labels=obstacle_labels, 
                                           drivable_labels=drivable_labels)
        time_elapsed = time.time() - tic
        print(f"Processed {file_name} in {time_elapsed:.4f} seconds")
        
        downsampled_visual = np.zeros((downsampled_map.shape[0], downsampled_map.shape[1], 3), dtype=np.uint8)
        # 将标签映射回 RGB 颜色
        for label, color in CityScapesPaletteDecode.items():
            downsampled_visual[downsampled_map == label] = color[::-1]  # ID to CityScapesPaletteDecode RGB
        downsampled_visual = cv2.resize(downsampled_visual, 
                                        (downsampled_visual.shape[1], downsampled_visual.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
        save_path = os.path.join(output_path, file_name)
        cv2.imwrite(save_path, downsampled_visual)

def process_folder(folder_path, output_path):
    """
    处理文件夹中的所有图像，并将结果保存到输出路径。

    参数：
    - folder_path: 输入图像文件夹路径
    - output_path: 输出结果文件夹路径
    """
    labels = [ele for ele in range(29)]
    drivable_labels = [1, 12]   # freespace, roadline
    city_obstacle_labels = [key for key, value in FreespaceIDMapping.items() if value not in drivable_labels]
    city_drivable_labels = [ele for ele in labels if ele not in city_obstacle_labels]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    file_list = os.listdir(folder_path)
    args_list = [
        (file_name, folder_path, output_path, city_obstacle_labels, city_drivable_labels)
        for file_name in file_list
        if file_name.endswith('.png')
    ]

    num_processes = cpu_count()  # 获取CPU核心数量
    with Pool(processes=num_processes) as pool:
        pool.map(process_image, args_list)

if __name__ == "__main__":
    input_folder = '/workspace/drWorkspace/BEVParkingOL/data/carla_bev/CAM_BEV_SEGMENTATION/'
    output_folder = '/workspace/drWorkspace/BEVParkingOL/data/carla_bev/CAM_BEV_SEGMENTATION_DOWN/'

    process_folder(input_folder, output_folder)
