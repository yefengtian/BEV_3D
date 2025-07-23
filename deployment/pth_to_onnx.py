import torch
import onnx
from your_project.model import build_bev_model

def export_bev_to_onnx(pth_path, onnx_out_path, batch_size=1, img_shape=(3, 256, 704)):
    model = build_bev_model()
    ckpt = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_img = torch.randn(batch_size, *img_shape, requires_grad=False)
    dummy_lidar = torch.randn(batch_size, 256, 256, requires_grad=False)  # 随实际输入修改

    print("[INFO] Exporting with inputs:", dummy_img.shape, dummy_lidar.shape)
    torch.onnx.export(
        model,
        (dummy_img, dummy_lidar),
        onnx_out_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["images", "lidar_feats"],
        output_names=["bev"],
        dynamic_axes={
            "images": {0: "batch", 2: "H", 3: "W"},
            "lidar_feats": {0: "batch"},
            "bev": {0: "batch"}
        },
        verbose=False
    )
    print(f"[SUCCESS] Saved ONNX model to {onnx_out_path}")

    onnx_model = onnx.load(onnx_out_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model is valid")

if __name__ == "__main__":
    export_bev_to_onnx(
        pth_path="checkpoints/best.pth",
        onnx_out_path="bevdepth_export.onnx",
        batch_size=1,
        img_shape=(3, 256, 704)
    )
