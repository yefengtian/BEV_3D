import cv2
import torch
import numpy as np

FreespacePalette = {
    0: (0, 0, 0),                # occ
    1: (255, 255, 255),          # no occ
}

ParkingspotType_decode = {
    0:"Perp",        1:"Para",       2:"Others"
}

ParkingspotStatus_decode = {
    0:"Vac",         1:"vehOcc",     2:"otherOcc"
}

pl_border_color = (30, 144, 255)    # RGB, dodgerblue
pl_entry_color = (139, 0, 0)        # RGB, darkred
text_color = (0, 0, 128)            # RGB, navy

def occ2img(semantics=None, target_size=(1600, 1600)):
    assert semantics is not None
    assert semantics.ndim == 3 and semantics.shape[-1] == 1 # (H, W, 1) for freespace currently

    # convert semantics to RGB image
    viz = np.zeros((semantics.shape[0], semantics.shape[1], 3), dtype=np.uint8)
    for i in range(len(FreespacePalette)):
        viz[semantics[..., 0] == i] = FreespacePalette[i][::-1]     # to BGR for cv2

    viz = viz[::-1, ::-1, ...]
    viz = cv2.resize(viz, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    return viz

def overlay_segmentation(base_img, seg_mask, color=[0, 255, 0], alpha=0.5):
    """
    将分割结果以透明度方式叠加在底图上
    
    Args:
        base_img: 底图 (BGR格式)
        seg_mask: 分割掩码 (二值图，0和255)
        color: 叠加颜色 [B, G, R]
        alpha: 透明度 (0-1)
    """
    # 创建彩色掩码
    color_mask = np.zeros_like(base_img)
    color_mask[seg_mask > 0] = color
    
    # 叠加图像
    overlay = base_img.copy()
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    return overlay

def draw_parkinglot(canvas_pred, pl_pred, bev_range=[-10, -10, 10, 10], target_size=(1600, 1600)):
    # bev_range: [xmin, ymin, xmax, ymax] in VCS
    target_width = target_size[0]
    target_height = target_size[1]

    pl_pred = [ele[0] for ele in pl_pred]   # task_id == 1
    centers, scores, labels, statuses, kpses = pl_pred
    parkingspot_num = len(centers)

    for i in range(parkingspot_num):
        center = centers[i].cpu().numpy()
        score = scores[i].cpu().numpy().astype(np.float)
        label = labels[i].cpu().numpy().astype(np.int)
        status = statuses[i].cpu().numpy().astype(np.int)

        kps = [kpses[idx][i].cpu().numpy().astype(np.float) for idx in range(4)]
        kps = np.array(kps).reshape(4, 2)

        # convert kps from vcs to bev_range
        kps_canvas = np.zeros_like(kps)
        kps_canvas[:, 1] = (bev_range[2] - kps[:, 0]) / (bev_range[2] - bev_range[0]) * target_height
        kps_canvas[:, 0] = (bev_range[3] - kps[:, 1]) / (bev_range[3] - bev_range[1]) * target_width
        kps_canvas = kps_canvas.astype(np.int32)
        cv2.polylines(canvas_pred, [kps_canvas], True, pl_border_color[::-1], 5)

        # mark entry border
        entry_onefourth_x = int(kps_canvas[0, 0] + 0.25*(kps_canvas[3, 0] - kps_canvas[0, 0]))
        entry_onefourth_y = int(kps_canvas[0, 1] + 0.25*(kps_canvas[3, 1] - kps_canvas[0, 1]))
        cv2.circle(canvas_pred, (entry_onefourth_x, entry_onefourth_y), 4, pl_entry_color[::-1], 8)

        # put text of label and status at the parkingspot center
        ctr_canvas_y = (bev_range[2] - center[0]) / (bev_range[2] - bev_range[0]) * target_height
        ctr_canvas_x = (bev_range[3] - center[1]) / (bev_range[3] - bev_range[1]) * target_width
        text = f"{ParkingspotType_decode[int(label)]} | {ParkingspotStatus_decode[int(status)]}"
        cv2.putText(canvas_pred, text,
                    (int(ctr_canvas_x-75), int(ctr_canvas_y-25)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[::-1], 2)
    return canvas_pred

def stitch_images_horizontal(img1, img2):
    """
    水平拼接两张图片
    """
    # 确保两张图片高度相同
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2:
        # 调整高度到较小值
        min_height = min(h1, h2)
        img1 = cv2.resize(img1, (int(w1 * min_height / h1), min_height))
        img2 = cv2.resize(img2, (int(w2 * min_height / h2), min_height))
    
    # 水平拼接
    stitched = np.hstack((img1, img2))
    return stitched

def visualize(occ_pred, pl_pred, save_path,ori_img):
    occ_pred = occ_pred[0]  # bs=1
    # pl_pred = pl_pred[0]
    
    sem_pred = occ_pred.cpu().numpy() if isinstance(occ_pred, torch.Tensor) else occ_pred
    canvas_pred = occ2img(semantics=sem_pred)
    # canvas_pred = np.zeros((1600,1600, 3), dtype=np.uint8)
    # canvas_pred = ori_img
    # canvas_pred = draw_parkinglot(canvas_pred, pl_pred)

    # canvas_pred = cv2.cvtColor(canvas_pred, cv2.COLOR_BGR2GRAY)

    # 二值化掩码（如果不是二值图）
    # _, seg_mask = cv2.threshold(canvas_pred, 127, 255, cv2.THRESH_BINARY)

    # 叠加分割结果（绿色，50%透明度）
    # result = overlay_segmentation(ori_img, seg_mask, color=[0, 255, 0], alpha=0.5)

    result = stitch_images_horizontal(ori_img,canvas_pred)

    cv2.imwrite(save_path, result)