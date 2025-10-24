import os
import torch
import argparse
import time
import cv2
from pathlib import Path

from mmcv import Config
from mmdet.utils import compat_cfg

from model_interface.model_interface import get_parking_model
from preprocess.preprocess import preprocess, PrepareParameter
from postprocess.vis import visualize
from dataset.offline_image_dataset import OfflineImageDataset
from utils.cam_params import params
from postprocess.eva_freespace import FreespaceEvaluator

def main():
    parser = argparse.ArgumentParser(description="Offline image inference script")
    parser.add_argument(
        "--config", type=str, default="model_interface/config/freespace_occ2d_r50_depth.py"
    )
    parser.add_argument(
        "--weights", type=str, default="only_freespace_head_1020_run/best_val_loss_epoch_035.pth"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/carla_bev", help="Path to image data directory"
    )
    parser.add_argument(
        "--annotation_file", type=str, default="data/carla_bev/0801_all_51725_test_20_samples.pkl", help="Path to annotation file (optional)"
    )
    parser.add_argument(
        "--vis", type=str, default=None, help="Directory for output visualization"
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

    evaluator = FreespaceEvaluator(boundary_tol_px = 2)
    for idx in range(start_idx, end_idx):
        try:
            all_data = dataset.get_data_info(idx)
            if not all_data:
                print(f"Skipping sample {idx}: no data")
                continue

            new_img_path = all_data['curr']['cams']['CAM_BEV_FREESPACE_BINARY']['data_path']
            # new_img_path = img_path.replace('data/carla_bev/','data/carla_bev_infer_data_fast/')
            print(new_img_path)
            img_name = os.path.basename(new_img_path)
            img = cv2.imread(new_img_path,cv2.IMREAD_UNCHANGED)

            tic = time.time()
            with torch.no_grad():
                infer_data = preprocess(args.config, all_data, input_params)
                toc1 = time.time()
                print(f"Sample {idx} - Time elapsed of preprocess: {int(1000*(toc1 - tic))}ms")
                
                occ_pred = model(return_loss=False, rescale=True, **infer_data)
                toc2 = time.time()
                print(f"Sample {idx} - Time elapsed of model: {int(1000*(toc2 - toc1))}ms")

                pred_arr = occ_pred[0]
                res_one = evaluator.update(pred_arr,img,pred_is_logit=False)
                print(f"[Eval] sample {idx}:{res_one}")

                pl_pred = None
                if args.vis is not None:
                    timestamp = all_data.get('timestamp', idx)
                    save_path = os.path.join(args.vis, img_name)
                    visualize(occ_pred, pl_pred, save_path,img)
                toc3 = time.time()
                print(f"Sample {idx} - Time elapsed of visualize: {int(1000*(toc3 - toc2))}ms")

            print(f"Sample {idx} - Total time elapsed: {int(1000*(toc3 - tic))}ms\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    summary = evaluator.summarize()
    print("[Eval][Summary]",summary)

if __name__ == "__main__":
    main() 