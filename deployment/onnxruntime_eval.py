import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("bevdepth_export.onnx")
ort_outs = sess.run(None, {
    "images": dummy_img.numpy(),
    "lidar_feats": dummy_lidar.numpy()
})
torch_out = model(dummy_img, dummy_lidar)
np.testing.assert_allclose(ort_outs[0], torch_out.detach().cpu().numpy(), rtol=1e-3, atol=1e-5)
print("✅ ONNX output matches PyTorch")
