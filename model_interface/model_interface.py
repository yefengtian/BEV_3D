import os
import sys
import torch
from tqdm import tqdm

from mmdet.apis import set_random_seed
from mmdet.utils import setup_multi_processes, compat_cfg
from mmdet3d.models import build_model
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

sys.path.insert(0, os.getcwd())


def get_parking_model(config=None, weights=None):
    if config is None:
        config = 'model_interface/config/freespace_occ2d_r50_depth.py'

    if weights is None:
        weights = 'model_interface/ckpts/epoch_300.pth'

    cfgs = Config.fromfile(config)
    cfgs = compat_cfg(cfgs)

    setup_multi_processes(cfgs)
    set_random_seed(0, deterministic=True)

    assert torch.cuda.is_available()

    model = build_model(cfgs.model).cuda()
    model = MMDataParallel(model, [0]).eval()

    load_checkpoint(model, weights, map_location='cuda', strict=True)

    return model

def main():
    model = get_parking_model()

if __name__ == '__main__':
    main()