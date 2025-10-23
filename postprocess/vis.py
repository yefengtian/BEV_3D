import cv2
import torch
import numpy as np

FreespacePalette = {
    0: (0, 0, 0),                # unlabeled, black
    1: (169, 169, 169),          # freespace, darkgray
    2: (0, 255, 255),            # sidewalks, aqua
    3: (100, 149, 237),          # building, cornflowerblue
    4: (255, 192, 203),          # fence, pink
    5: (255, 255, 0),            # pole, yellow
    6: (189, 183, 107),          # terrain, darkkhaki
    7: (255, 0, 255),            # pedestrian, fuscia
    8: (123, 104, 238),          # rider, mediumslateblue
    9: (0, 255, 0),              # vehicle, lime
    10: (0, 128, 0),             # train, green
    11: (160, 82, 45),           # others, sienna
    12: (255, 250, 250)          # roadline, snow
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


# 颜色(BGR)
OCC2COLOR = {
   1: (60, 200, 60),   # vacant -> 绿
   2: (30, 30, 220),   # VehOcc -> 红
   3: (0, 165, 255)    # OtherOcc -> 橙
}
def _ego_to_img_xy(x, y, bev_range, scale_x, scale_y):
   """
   车辆坐标(x前,y左) -> 图像像素(u右,v下)。为使“上=前”，v 用 y_max - y。
   bev_range = [xmin, ymin, xmax, ymax] (单位: m)
   """
   xmin, ymin, xmax, ymax = bev_range
   v = (xmax - x) * scale_x
   u = (ymax - y) * scale_y
   return int(round(u)), int(round(v))
def _draw_dashed(img, p0, p1, color, thickness=2, dash=12, gap=8):
   p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
   v = p1 - p0; L = np.linalg.norm(v)
   if L < 1e-6: return
   d = v / L; t = 0.0
   while t < L:
       a = p0 + d * t
       b = p0 + d * min(L, t + dash)
       cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                color, thickness, cv2.LINE_AA)
       t += dash + gap
def draw_slots_on_bev(pl_gt, canvas, bev_range=(-10,-10,10,10), target_size=(1600,1600),alpha_fill=0.25,
                     line_thickness=2, draw_ids=True):
    """
    pl_gt: list of dicts:
        { 'pl_category':1/2, 'occupied_category':1/2/3, 'id':int,
        'kps_in_ego': [[x,y,...], ...] }
    canvas: HxWx3 uint8 图像(可为None)，函数会统一resize为 target_size
    """
    W, H = int(target_size[0]), int(target_size[1])
    if canvas is None:
        img = np.full((H, W, 3), 30, np.uint8)     # 深灰底
    else:
        img = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_NEAREST)
    xmin, ymin, xmax, ymax = bev_range
    scale_x = W / float(xmax - xmin)
    scale_y = H / float(ymax - ymin)
    for slot in pl_gt:
        kps = slot.get('kps_in_ego', [])
        if len(kps) < 3:  # 至少三点
            continue
        kps_canvas = np.zeros_like(np.array(kps)[:,:2])

        # 映射到像素坐标（仅取xy）
        for ind_x,p in enumerate(kps):
            u, v = _ego_to_img_xy(float(p[0]), float(p[1]),
                                    bev_range, scale_x, scale_y)
            kps_canvas[ind_x, 0] = u
            kps_canvas[ind_x, 1] = v
            # pts = np.array(pts, dtype=np.int32)
            # color = OCC2COLOR.get(int(slot.get('occupied_category', 1)), (200,200,200))color = ()
        


        kps_canvas = kps_canvas.astype(np.int32)
        center = [(kps_canvas[0,0]+kps_canvas[2,0])/2,(kps_canvas[0,1]+kps_canvas[2,1])/2]
        cv2.polylines(img, [kps_canvas], True, pl_border_color[::-1], 5)

        # mark entry border
        entry_onefourth_x = int(kps_canvas[0, 0] + 0.25*(kps_canvas[3, 0] - kps_canvas[0, 0]))
        entry_onefourth_y = int(kps_canvas[0, 1] + 0.25*(kps_canvas[3, 1] - kps_canvas[0, 1]))
        cv2.circle(img, (entry_onefourth_x, entry_onefourth_y), 4, pl_entry_color[::-1], 8)

        # put text of label and status at the parkingspot center
        label = slot["pl_category"]
        status = slot["occupied_category"]
        if status >=3:
            status = 3
        ctr_canvas_y = center[1]
        ctr_canvas_x = center[0]
        text = f"{ParkingspotType_decode[int(label)-1]} | {ParkingspotStatus_decode[int(status)-1]}"
        cv2.putText(img, text,
                    (int(ctr_canvas_x-75), int(ctr_canvas_y-25)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[::-1], 2)

    return img

def visualize(occ_pred, pl_pred, save_path,ori_img,pl_gt = None):
    # occ_pred = occ_pred[0]  # bs=1
    pl_pred = pl_pred[0]
    
    # sem_pred = occ_pred.cpu().numpy() if isinstance(occ_pred, torch.Tensor) else occ_pred
    # canvas_pred = occ2img(semantics=sem_pred)
    # canvas_pred = np.zeros((1600,1600, 3), dtype=np.uint8)
    canvas_pred = ori_img
    if pl_gt is not None:
        gt_vis = draw_slots_on_bev(pl_gt, canvas_pred)
    canvas_pred = draw_parkinglot(canvas_pred, pl_pred)

    result = stitch_images_horizontal(ori_img,canvas_pred)
    cv2.imwrite(save_path, result)