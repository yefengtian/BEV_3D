import cv2
import numpy as np
from dataclasses import dataclass, field
# ---------- 基础：尺寸与二值化 ----------
def to_binary_mask(arr, pred_is_logit=False, thr=0.5):
   """
   兼容 torch.Tensor / np.ndarray，形状 (H,W) 或 (H,W,1) 或 [1,H,W]。
   pred_is_logit=True：arr 可为概率[0,1]或logit(任意实数)；自动做sigmoid。
   """
   try:
       import torch
       is_torch = isinstance(arr, torch.Tensor)
   except ImportError:
       is_torch = False
   if is_torch:
       # squeeze 到 (H,W)
       if arr.dim() == 3 and arr.shape[-1] == 1:
           arr = arr[..., 0]
       if arr.dim() == 3 and arr.shape[0] == 1:
           arr = arr[0]
       x = arr.float()
       if pred_is_logit:
           # 若已是概率也没关系；sigmoid 在[0,1]区间近似恒等
           x = torch.sigmoid(x)
           mask = (x >= thr).to(torch.uint8)
       else:
           # 已是离散 0/1（或0/255），统一阈值>0
           mask = (x > 0).to(torch.uint8)
       # 返回 numpy，后续评估都用 numpy
       return mask.detach().cpu().numpy()
   else:
       # numpy 分支
       if arr.ndim == 3 and arr.shape[-1] == 1:
           arr = arr[..., 0]
       if arr.ndim == 3 and arr.shape[0] == 1:
           arr = arr[0]
       x = arr.astype(np.float32)
       if pred_is_logit:
           if x.max() > 1.0 or x.min() < 0.0:
               x = 1.0 / (1.0 + np.exp(-x))  # sigmoid
           mask = (x >= thr).astype(np.uint8)
       else:
           mask = (x > 0).astype(np.uint8)
       return mask
def binarize_gt(gt_img, invert=False):
   """
   gt_img: 读出来的 0/255 图（H×W 或 H×W×C）——非零记为占用=1。
   invert=True 时，交换 free/occupied 的定义（少见，用于数据标注相反时）。
   """
   if gt_img.ndim == 3:
       gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
   else:
       gt_gray = gt_img
   gt = (gt_gray > 127).astype(np.uint8)
   if invert:
       gt = 1 - gt
   return gt
def ensure_same_size(mask, target_hw):
   Ht, Wt = target_hw
   if mask.shape[0] != Ht or mask.shape[1] != Wt:
       mask = cv2.resize(mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
   return mask
# ---------- 像素级统计 ----------
def confusion_binary(pred, gt):
   """
   返回 TP, FP, FN, TN (以 occupied=1 为正类)
   """
   assert pred.shape == gt.shape
   pred = pred.astype(np.uint8)
   gt   = gt.astype(np.uint8)
   TP = np.sum((pred == 1) & (gt == 1))
   FP = np.sum((pred == 1) & (gt == 0))
   FN = np.sum((pred == 0) & (gt == 1))
   TN = np.sum((pred == 0) & (gt == 0))
   return TP, FP, FN, TN
def iou_dice_from_confusion(TP, FP, FN, eps=1e-6):
   iou  = TP / (TP + FP + FN + eps)
   dice = 2*TP / (2*TP + FP + FN + eps)
   return iou, dice
def per_class_iou_dice(pred, gt, eps=1e-6):
   """
   返回 IoU/Dice of occupied(1) 和 free(0) 以及 mIoU/meanDice
   """
   # 对正类(1)
   TP1, FP1, FN1, TN1 = confusion_binary(pred, gt)
   iou_occ, dice_occ   = iou_dice_from_confusion(TP1, FP1, FN1, eps)
   # 对负类(0)：把 0 当正类再算一遍
   pred_inv = 1 - pred
   gt_inv   = 1 - gt
   TP0, FP0, FN0, TN0 = confusion_binary(pred_inv, gt_inv)
   iou_free, dice_free = iou_dice_from_confusion(TP0, FP0, FN0, eps)
   miou      = 0.5 * (iou_occ + iou_free)
   mean_dice = 0.5 * (dice_occ + dice_free)
   return {
       "IoU/occupied": float(iou_occ),
       "IoU/free":     float(iou_free),
       "mIoU":         float(miou),
       "Dice/occupied":float(dice_occ),
       "Dice/free":    float(dice_free),
       "MeanDice":     float(mean_dice),
   }
def precision_recall_f1(pred, gt, eps=1e-6):
   TP, FP, FN, TN = confusion_binary(pred, gt)
   prec = TP / (TP + FP + eps)
   rec  = TP / (TP + FN + eps)
   f1   = 2*prec*rec / (prec + rec + eps)
   acc  = (TP + TN) / (TP + FP + FN + TN + eps)
   # 安全相关：漏检率(FN-rate on occupied) 与 误报率(FP-rate on free)
   occ_pixels  = TP + FN
   free_pixels = TN + FP
   fn_rate_on_occ  = FN / (occ_pixels + eps)
   fp_rate_on_free = FP / (free_pixels + eps)
   return {
       "Precision(occ)": float(prec),
       "Recall(occ)":    float(rec),
       "F1(occ)":        float(f1),
       "PixelAcc":       float(acc),
       "FN-rate@occupied": float(fn_rate_on_occ),
       "FP-rate@free":     float(fp_rate_on_free),
   }
# ---------- 边界质量 ----------
def boundary_map(mask):
   """提取二值 mask 的边界（单像素）"""
   mask = mask.astype(np.uint8)
   edges = cv2.Canny(mask*255, 0, 1)  # 低阈值，取轮廓
   edges = (edges > 0).astype(np.uint8)
   return edges
def boundary_f1(pred, gt, tolerance_px=2, eps=1e-6):
   """
   Boundary-F1 with tolerance: 计算 pred 与 gt 边界的相互匹配率
   """
   be_pred = boundary_map(pred)
   be_gt   = boundary_map(gt)
   if tolerance_px > 0:
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance_px+1, 2*tolerance_px+1))
       be_gt_dil   = cv2.dilate(be_gt, kernel)
       be_pred_dil = cv2.dilate(be_pred, kernel)
   else:
       be_gt_dil, be_pred_dil = be_gt, be_pred
   # precision: pred 边界有多少能在 gt 容差膨胀内匹配
   tp_p = np.sum((be_pred == 1) & (be_gt_dil == 1))
   pp   = np.sum(be_pred == 1)
   precision_b = tp_p / (pp + eps)
   # recall: gt 边界有多少能在 pred 容差膨胀内匹配
   tp_r = np.sum((be_gt == 1) & (be_pred_dil == 1))
   gp   = np.sum(be_gt == 1)
   recall_b = tp_r / (gp + eps)
   f1_b = 2 * precision_b * recall_b / (precision_b + recall_b + eps)
   return {
       f"BoundaryF1@{tolerance_px}px": float(f1_b),
       f"BoundaryPrecision@{tolerance_px}px": float(precision_b),
       f"BoundaryRecall@{tolerance_px}px": float(recall_b),
   }
# ---------- 距离型（平均表面距离，双向） ----------
def mean_surface_distance(pred, gt):
   """
   使用距离变换计算边界到对方边界的平均距离（像素）。
   返回双向平均： (d(pred->gt) + d(gt->pred)) / 2
   """
   be_pred = boundary_map(pred)
   be_gt   = boundary_map(gt)
   # 距离变换对“0”做距离，所以取反
   dist_to_gt   = cv2.distanceTransform((1 - be_gt).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
   dist_to_pred = cv2.distanceTransform((1 - be_pred).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
   pred_to_gt = dist_to_gt[be_pred == 1]
   gt_to_pred = dist_to_pred[be_gt == 1]
   d1 = float(pred_to_gt.mean()) if pred_to_gt.size > 0 else 0.0
   d2 = float(gt_to_pred.mean()) if gt_to_pred.size > 0 else 0.0
   return {
       "MSD(pred→gt)": d1,
       "MSD(gt→pred)": d2,
       "MSD(mean)":    (d1 + d2) / 2.0
   }
# ---------- 单张样本评估入口 ----------
def evaluate_freespace_sample(pred_arr, gt_img,
                             pred_is_logit=False, thr=0.5,
                             match_size_to="pred",  # "pred" 或 "gt"
                             invert_gt=False,
                             boundary_tol_px=2):
   """
   pred_arr: 预测（np.ndarray 或 torch.Tensor），类别图或概率/logit
   gt_img:   读入的 GT 图像（0/255）
   match_size_to: 把另一方 resize 到这方的尺寸，保持最近邻
   """
   # 兼容 torch.Tensor
   try:
       import torch
       if isinstance(pred_arr, torch.Tensor):
           pred_arr = pred_arr.detach().cpu().numpy()
   except ImportError:
       pass
   # squeeze 到 (H,W)
   if pred_arr.ndim == 3 and pred_arr.shape[-1] == 1:
       pred_arr = pred_arr[..., 0]
   if pred_arr.ndim == 3 and pred_arr.shape[0] == 1:  # [1,H,W]
       pred_arr = pred_arr[0]
   if pred_arr.ndim == 3 and pred_arr.shape[0] != 1 and pred_arr.shape[-1] != 1:
       raise ValueError("pred_arr 形状异常，请提供 (H,W) 或 (H,W,1) 或 [1,H,W]")
   # 二值化
   pred_mask = to_binary_mask(pred_arr, pred_is_logit=pred_is_logit, thr=thr)
   gt_mask   = binarize_gt(gt_img, invert=invert_gt)
   # 尺寸对齐
   if match_size_to == "pred":
       gt_mask = ensure_same_size(gt_mask, pred_mask.shape[:2])
   else:
       pred_mask = ensure_same_size(pred_mask, gt_mask.shape[:2])
   # 指标计算
   out = {}
   out.update(per_class_iou_dice(pred_mask, gt_mask))
   out.update(precision_recall_f1(pred_mask, gt_mask))
   out.update(boundary_f1(pred_mask, gt_mask, tolerance_px=boundary_tol_px))
   out.update(mean_surface_distance(pred_mask, gt_mask))
   return out
# ---------- 汇总器（多样本聚合） ----------
@dataclass
class FreespaceEvaluator:
   boundary_tol_px: int = 2
   n: int = 0
   # 用“总和”聚合可稳定 mIoU/Dice 等（micro-averaging）
   TP1: int = 0; FP1: int = 0; FN1: int = 0; TN1: int = 0
   TP0: int = 0; FP0: int = 0; FN0: int = 0; TN0: int = 0
   sum_boundary_prec: float = 0.0
   sum_boundary_recall: float = 0.0
   sum_msd_mean: float = 0.0
   def update(self, pred_arr, gt_img, pred_is_logit=False, thr=0.5, match_size_to="pred", invert_gt=False):
       res = evaluate_freespace_sample(pred_arr, gt_img,
                                       pred_is_logit=pred_is_logit, thr=thr,
                                       match_size_to=match_size_to, invert_gt=invert_gt,
                                       boundary_tol_px=self.boundary_tol_px)
       # micro confusion（按 occupied 正类）
       # 重新计算一次混淆用于微平均（避免逐样本取均值的偏置）
       pred_mask = to_binary_mask(pred_arr, pred_is_logit=pred_is_logit, thr=thr)
       gt_mask   = binarize_gt(gt_img, invert=invert_gt)
       if match_size_to == "pred":
           gt_mask = ensure_same_size(gt_mask, pred_mask.shape[:2])
       else:
           pred_mask = ensure_same_size(pred_mask, gt_mask.shape[:2])
       TP1, FP1, FN1, TN1 = confusion_binary(pred_mask, gt_mask)
       self.TP1 += TP1; self.FP1 += FP1; self.FN1 += FN1; self.TN1 += TN1
       pred_inv = 1 - pred_mask; gt_inv = 1 - gt_mask
       TP0, FP0, FN0, TN0 = confusion_binary(pred_inv, gt_inv)
       self.TP0 += TP0; self.FP0 += FP0; self.FN0 += FN0; self.TN0 += TN0
       self.sum_boundary_prec  += res[f"BoundaryPrecision@{self.boundary_tol_px}px"]
       self.sum_boundary_recall+= res[f"BoundaryRecall@{self.boundary_tol_px}px"]
       self.sum_msd_mean       += res["MSD(mean)"]
       self.n += 1
       return res  # 返回当前样本指标
   def summarize(self):
       eps = 1e-6
       # occupied 正类
       iou_occ, dice_occ = iou_dice_from_confusion(self.TP1, self.FP1, self.FN1, eps)
       # free 当正类
       iou_free, dice_free = iou_dice_from_confusion(self.TP0, self.FP0, self.FN0, eps)
       miou      = 0.5*(iou_occ + iou_free)
       mean_dice = 0.5*(dice_occ + dice_free)
       # Precision/Recall/F1/Acc（以 occupied 为正）
       TP, FP, FN, TN = self.TP1, self.FP1, self.FN1, self.TN1
       prec = TP / (TP + FP + eps)
       rec  = TP / (TP + FN + eps)
       f1   = 2*prec*rec / (prec + rec + eps)
       acc  = (TP + TN) / (TP + FP + FN + TN + eps)
       occ_pixels  = TP + FN
       free_pixels = self.TN0 + self.FP0  # 等价于 TN+FP
       fn_rate_on_occ  = FN / (occ_pixels + eps)
       fp_rate_on_free = FP / (free_pixels + eps)
       return {
           "IoU/occupied": float(iou_occ),
           "IoU/free":     float(iou_free),
           "mIoU":         float(miou),
           "Dice/occupied":float(dice_occ),
           "Dice/free":    float(dice_free),
           "MeanDice":     float(mean_dice),
           "Precision(occ)": float(prec),
           "Recall(occ)":    float(rec),
           "F1(occ)":        float(f1),
           "PixelAcc":       float(acc),
           "FN-rate@occupied": float(fn_rate_on_occ),
           "FP-rate@free":     float(fp_rate_on_free),
           f"BoundaryPrecision@{self.boundary_tol_px}px": float(self.sum_boundary_prec / max(1,self.n)),
           f"BoundaryRecall@{self.boundary_tol_px}px":    float(self.sum_boundary_recall / max(1,self.n)),
           f"BoundaryF1@{self.boundary_tol_px}px":        float( 2*(self.sum_boundary_prec/self.n)*(self.sum_boundary_recall/self.n) /
                                                                 (self.sum_boundary_prec/self.n + self.sum_boundary_recall/self.n + 1e-6) ),
           "MSD(mean)": float(self.sum_msd_mean / max(1,self.n)),
           "Samples":   int(self.n)
       }