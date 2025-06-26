import os
import torch
import argparse
import pickle
import time

from mmcv import Config
from mmdet.utils import compat_cfg

from utils.redis_utils import RedisHelper
from utils.cam_params import params

from model_interface.model_interface import get_parking_model
from preprocess.preprocess import preprocess, PrepareParameter
from postprocess.vis import visualize

# from config import transform_configs
# from postprocess import freespace
# from postprocess import parkinglot

# def postprocess():
#     freespace
#     parkinglot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--config", type=str, default="model_interface/config/freespace_occ2d_r50_depth.py"
    )
    parser.add_argument(
        "--weights", type=str, default="model_interface/ckpts/epoch_69.pth"
    )
    parser.add_argument(
        "--vis", type=str, default="./vis", help="Directory for output visualization, None for no visualization"
    )
    args = parser.parse_args()

    if args.vis is not None:
        os.makedirs(args.vis, exist_ok=True)

    subscriber = RedisHelper().subscribe("mmdet_data")
    
    cfgs = compat_cfg(Config.fromfile(args.config))
    input_params = PrepareParameter(cfgs._cfg_dict["data_config"], params).get_inputs()
    model = get_parking_model(args.config, args.weights)

    while 1:
        message = subscriber.parse_response()
        if not message:
            continue
        
        tic = time.time()
        
        
        with torch.no_grad():
            pickled_data = pickle.loads(message[2])
            toc1 = time.time()
            print(f"Time elapsed of pickle.loads: {int(1000*(toc1 - tic))}ms")
            
            infer_data = preprocess(args.config, pickled_data, input_params)
            toc2 = time.time()
            print(f"Time elapsed of preprocess: {int(1000*(toc2 - toc1))}ms")
            
            occ_pred, pl_pred = model(return_loss=False, rescale=True, **infer_data)
            toc3 = time.time()
            print(f"Time elapsed of model: {int(1000*(toc3 - toc2))}ms")
            
            if args.vis is not None:
                save_path = os.path.join(args.vis, f"{pickled_data['timestamp']}.jpg")
                visualize(occ_pred, pl_pred, save_path)
            toc4 = time.time()
            print(f"Time elapsed of visualize: {int(1000*(toc4 - toc3))}ms")

        print(f"Time elapsed of {pickled_data['timestamp']}: {int(1000*(toc4 - tic))}ms\n")

    # print(model)
