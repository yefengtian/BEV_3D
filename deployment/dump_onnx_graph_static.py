import os
import torch
import onnx
from argparse import ArgumentParser
from mmdet3d.models.necks.view_transformer import ViewTransformer

# 是否导出ONNX模板标志
exporting_to_onnx = True

# 替代 quickcumsum 的 scatter_add 实现
def bev_pool_v2_scatter(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                        bev_feat_shape, interval_starts, interval_lengths):
    B, Dz, Dy, Dx, C = bev_feat_shape
    feat_pts = feat.reshape(-1, feat.shape[-1])
    xv = torch.zeros((B * Dz * Dy * Dx, C),
                     device=feat_pts.device, dtype=feat_pts.dtype)
    xv = xv.scatter_add(0, ranks_bev.unsqueeze(1).expand(-1, C), feat_pts)
    x = xv.view(B, Dz, Dy, Dx, C)
    return x.permute(0, 4, 1, 2, 3).contiguous()

# 保存原方法
_original_voxel_pool = ViewTransformer.voxel_pooling_v2

# patch 方法
def patched_voxel_pooling_v2(self, coor, depth, feat):
    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
        self.voxel_pooling_prepare_v2(coor)
    if ranks_feat is None:
        return _original_voxel_pool(self, coor, depth, feat)
    feat2 = feat.permute(0, 1, 3, 4, 2)
    bev_feat_shape = (depth.shape[0],
                      int(self.grid_size[2]),
                      int(self.grid_size[1]),
                      int(self.grid_size[0]),
                      feat2.shape[-1])
    if exporting_to_onnx:
        bev_feat = bev_pool_v2_scatter(depth, feat2,
                                       ranks_depth, ranks_feat,
                                       ranks_bev,
                                       bev_feat_shape,
                                       interval_starts, interval_lengths)
    else:
        bev_feat = _original_voxel_pool(self, coor, depth, feat2)
    if self.collapse_z:
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
    return bev_feat

# 应用 patch
ViewTransformer.voxel_pooling_v2 = patched_voxel_pooling_v2

# Export 包装器，调用 forward_test
class BEVExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, imgs, sensor2egos, ego2globals,
                intrins, post_rots, post_trans, bda_rot):
        single_input = [imgs, sensor2egos, ego2globals, intrins,
                        post_rots, post_trans, bda_rot]
        img_inputs = [single_input]  # 原先你的代码是这样嵌套
        img_metas = [{
            'box_type_3d': None,
            'box_mode_3d': None
        }]
        # 调用 original forward_test 接口
        return self.model.forward_test(
            img_inputs=img_inputs, img_metas=img_metas)[0]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--onnx_path', default='tmp_model.onnx')
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()

def dump_onnx_graph(cfg, pth_path, device='cpu', onnx_path='tmp.onnx'):
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(cfg)
    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    ckpt = torch.load(pth_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()

    wrapped = BEVExportWrapper(model).to(device)

    B, N, H, W = 1, cfg.data.test.pipeline[0].N_views, cfg.data.test.img_h, cfg.data.test.img_w
    dummy_imgs = torch.randn(B, N, 3, H, W, device=device)
    dummy_s2e = torch.eye(4).repeat(B, N, 1, 1).to(device)
    dummy_e2g = torch.eye(4).repeat(B, N, 1, 1).to(device)
    dummy_intr = torch.eye(3).repeat(B, N, 1, 1).to(device)
    dummy_post_rots = torch.eye(3).repeat(B, N, 1, 1).to(device)
    dummy_post_trans = torch.zeros(B, N, 3).to(device)
    dummy_bda_rot = torch.eye(3).to(device)

    torch.onnx.export(
        wrapped,
        (dummy_imgs, dummy_s2e, dummy_e2g,
         dummy_intr, dummy_post_rots, dummy_post_trans, dummy_bda_rot),
        onnx_path,
        opset_version=13,
        input_names=[
            "imgs", "sensor2egos", "ego2globals",
            "intrins", "post_rots", "post_trans", "bda_rot"
        ],
        output_names=["bev_output"],
        dynamic_axes={
            "imgs": {0: "batch", 1: "num_views"},
            "bev_output": {0: "batch"}
        }
    )

    print("Exported ONNX model to:", onnx_path)
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model is valid!")

if __name__ == '__main__':
    args = parse_args()
    dump_onnx_graph(args.config, args.checkpoint,
                    args.device, args.onnx_path)
