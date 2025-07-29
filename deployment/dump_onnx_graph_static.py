import os
import torch
import onnx

# 标志位：是否 ONNX 导出模式
exporting_to_onnx = True

# 导出用的替代函数
def bev_pool_v2_scatter(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                        bev_feat_shape, interval_starts, interval_lengths):
    B, Dz, Dy, Dx, C = bev_feat_shape
    # 展平点特征
    feat_pts = feat.reshape(-1, feat.shape[-1])
    N_pts = ranks_bev.numel()
    # 创建输出
    xv = torch.zeros((B * Dz * Dy * Dx, C), device=feat_pts.device, dtype=feat_pts.dtype)
    # scatter_add
    xv = xv.scatter_add(0, ranks_bev.unsqueeze(1).expand(-1, C), feat_pts)
    x = xv.view(B, Dz, Dy, Dx, C)
    return x.permute(0, 4, 1, 2, 3).contiguous()

# patch 方法注入
from mmdet3d.models.necks.view_transformer import ViewTransformer  # 或你的类路径
original_voxel_pool = ViewTransformer.voxel_pooling_v2

def patched_voxel_pooling_v2(self, coor, depth, feat):
    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
        self.voxel_pooling_prepare_v2(coor)
    if ranks_feat is None:
        return original_voxel_pool(self, coor, depth, feat)
    feat = feat.permute(0,1,3,4,2)
    bev_feat_shape = (depth.shape[0],
                      int(self.grid_size[2]),
                      int(self.grid_size[1]),
                      int(self.grid_size[0]),
                      feat.shape[-1])
    if exporting_to_onnx:
        bev_feat = bev_pool_v2_scatter(depth, feat, ranks_depth,
                                       ranks_feat, ranks_bev,
                                       bev_feat_shape,
                                       interval_starts, interval_lengths)
    else:
        bev_feat = original_voxel_pool(self, coor, depth, feat)
    if self.collapse_z:
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
    return bev_feat

ViewTransformer.voxel_pooling_v2 = patched_voxel_pooling_v2

# ---------------------
# 下面是 ONNX 导出流程段（根据你的模型替换）
# ---------------------
if __name__ == '__main__':
    model = ...  # 导入/构造你的 mmdet3d 模型
    model.eval()
    dummy_input = ...  # 构造模型 forward 输入
    torch.onnx.export(
        model,
        dummy_input,
        "model_export.onnx",
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={ "input": {0:"batch",2:"h",3:"w"},
                       "output": {0:"batch"} }
    )
    print("ONNX model exported: model_export.onnx")

    # 测试加载
    onnx_model = onnx.load("model_export.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
