import os
import torch
import argparse
import time
from pathlib import Path

from mmcv import Config
from mmdet.utils import compat_cfg

from model_interface.model_interface import get_parking_model
from preprocess.preprocess import preprocess, PrepareParameter
from postprocess.vis import visualize
from dataset.offline_image_dataset import OfflineImageDataset
from utils.cam_params import params

def main():
    parser = argparse.ArgumentParser(description="Offline image inference script")
    parser.add_argument(
        "--config", type=str, default="model_interface/config/freespace_occ2d_r50_depth.py"
    )
    parser.add_argument(
        "--weights", type=str, default="model_interface/ckpts/epoch_69.pth"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to image data directory"
    )
    parser.add_argument(
        "--annotation_file", type=str, default=None, help="Path to annotation file (optional)"
    )
    parser.add_argument(
        "--vis", type=str, default="./vis", help="Directory for output visualization"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Start index for processing"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for processing"
    )
    args = parser.parse_args()

    if args.vis is not None:
        os.makedirs(args.vis, exist_ok=True)

    # 加载数据集
    dataset = OfflineImageDataset(
        data_root=args.data_root,
        annotation_file=args.annotation_file,
        test_mode=True
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # 加载模型
    cfgs = compat_cfg(Config.fromfile(args.config))
    input_params = PrepareParameter(cfgs._cfg_dict["data_config"], params).get_inputs()
    model = get_parking_model(args.config, args.weights)
    
    print("Model loaded successfully")

    # 设置处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    
    print(f"Processing samples from {start_idx} to {end_idx}")

    for idx in range(start_idx, end_idx):
        try:
            all_data = dataset.get_data_info(idx)
            if not all_data:
                print(f"Skipping sample {idx}: no data")
                continue
            
            tic = time.time()
            
            with torch.no_grad():
                infer_data = preprocess(args.config, all_data, input_params)
                toc1 = time.time()
                print(f"Sample {idx} - Time elapsed of preprocess: {int(1000*(toc1 - tic))}ms")
                
                occ_pred, pl_pred = model(return_loss=False, rescale=True, **infer_data)
                toc2 = time.time()
                print(f"Sample {idx} - Time elapsed of model: {int(1000*(toc2 - toc1))}ms")
                
                if args.vis is not None:
                    timestamp = all_data.get('timestamp', idx)
                    save_path = os.path.join(args.vis, f"{timestamp}.jpg")
                    visualize(occ_pred, pl_pred, save_path)
                toc3 = time.time()
                print(f"Sample {idx} - Time elapsed of visualize: {int(1000*(toc3 - toc2))}ms")

            print(f"Sample {idx} - Total time elapsed: {int(1000*(toc3 - tic))}ms\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

if __name__ == "__main__":
    main() 