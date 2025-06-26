from utils.redis_utils import RedisHelper
import mmcv 
from  dataset.carla_dataset import CarlaDataset
import pickle

red = RedisHelper()
ann_file="data/carla_bev/test_300.pkl"
Carla_data = CarlaDataset(ann_file = ann_file,
        test_mode=True,
        use_valid_flag=True,
        box_type_3d='LiDAR')

while 1:
    len_data = Carla_data.load_annotations(ann_file = ann_file)
    for index in range(len_data):
        all_data = Carla_data.get_data_info(index)
        if all_data:
            pickle_data = pickle.dumps(all_data)
            red.publish("mmdet_data", pickle_data)
            print(all_data["timestamp"])
    print("test data published!!!")