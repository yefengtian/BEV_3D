import torch
import onnx
from your_project.model import build_bev_model  # 修改为你的实际导入

def dump_onnx_graph(pth_path, dummy_input, onnx_path="tmp_model.onnx"):
    model = build_bev_model()
    ckpt = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    model.eval()

    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=13,
        input_names=["images", "lidar_feats"],  # 按模型实际输入命名
        output_names=["bev_output"],
        dynamic_axes={
            "images": {0: "B", 2: "H", 3: "W"},
            "lidar_feats": {0: "B"},
            "bev_output": {0: "B"},
        },
        verbose=True  # 会打印模型节点信息
    )

    onnx_model = onnx.load(onnx_path)
    print(onnx.helper.printable_graph(onnx_model.graph))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
