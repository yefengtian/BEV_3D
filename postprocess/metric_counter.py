# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import defaultdict

# ===== 标签定义（统一到0-based）=====
CAT_ID2NAME = {0: "Perp", 1: "Para", 2: "Others"}
STAT_ID2NAME = {0: "Vac", 1: "vehOcc", 2: "otherOcc"}

# ========== 工具函数 ==========
def _to_np(a):
    if hasattr(a, "detach"):  # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a)

def decode_pl_pred(pl_pred_item):
    """
    pl_pred_item: model输出的单张图片的pl结果（即你 visualize 里取的 pl_pred[0]）
      结构： [ele[0] for ele in pl_pred] -> centers, scores, labels, statuses, kpses
      - centers: (N,2) tensor
      - scores: (N,) tensor
      - labels: (N,) tensor, 0-based: {0:Perp,1:Para,2:Others}
      - statuses: (N,) tensor, 0-based: {0:Vac,1:vehOcc,2:otherOcc}
      - kpses: list长度4，每个是(N,2)，四个角点（顺序与训练一致）
    返回 list[dict]: [{'center':(2,), 'score':float, 'cat':int, 'stat':int, 'kps':(4,2)}]
    """
    centers, scores, labels, statuses, kpses = pl_pred_item  # each is tensor
    C = _to_np(centers)           # (N,2)
    S = _to_np(scores).reshape(-1)
    L = _to_np(labels).reshape(-1).astype(int)
    ST = _to_np(statuses).reshape(-1).astype(int)
    # kpses: list of 4 tensors (N,2)
    K = np.stack([_to_np(k) for k in kpses], axis=1)  # (N,4,2)

    N = C.shape[0]
    out = []
    for i in range(N):
        out.append({
            "center": C[i],
            "score": float(S[i]),
            "cat": int(L[i]),
            "stat": int(ST[i]),
            "kps": K[i].astype(float)  # (4,2) in ego (meters)
        })
    return out

def decode_pl_gt(pl_gt_item):
    """
    pl_gt_item: 数据集中当前帧GT格式（list of dict）
      dict: { 'pl_category':1/2/3, 'occupied_category':1/2/3, 'kps_in_ego': [[x,y], ...] (>=4点，取前4点) }
    统一转为0-based：cat-1, stat-1；并只取前4个点（若>4你可按需要重排）
    返回 list[dict]: [{'cat':int, 'stat':int, 'kps':(4,2)}]
    """
    out = []
    for s in pl_gt_item:
        cat = int(s.get('pl_category', 1)) - 1
        stat = int(s.get('occupied_category', 1)) - 1
        kps = np.asarray(s.get('kps_in_ego', []), dtype=float)
        if kps.shape[0] < 4:
            continue
        kps = kps[:4, :2]  # (4,2)
        out.append({
            "cat": max(0, min(2, cat)),
            "stat": max(0, min(2, stat)),
            "kps": kps
        })
    return out

def polygon_iou_quad(ptsA, ptsB):
    """
    IoU for convex quads using OpenCV.
    ptsA, ptsB: (4,2) float, 顺序为模型/标注约定的一致顺时针/逆时针
    """
    A = np.asarray(ptsA, dtype=np.float32)
    B = np.asarray(ptsB, dtype=np.float32)
    areaA = cv2.contourArea(A)
    areaB = cv2.contourArea(B)
    if areaA <= 1e-9 and areaB <= 1e-9:
        return 1.0
    if areaA <= 1e-9 or areaB <= 1e-9:
        return 0.0
    inter_area, inter_poly = cv2.intersectConvexConvex(A, B)
    union = areaA + areaB - inter_area
    if union <= 1e-9:
        return 0.0
    return float(inter_area / union)

def kp_dist_metrics(pred_kps, gt_kps):
    """
    两个四边形的对应关键点距离统计：max距离、mean距离
    """
    d = np.linalg.norm(np.asarray(pred_kps) - np.asarray(gt_kps), axis=1)  # (4,)
    return float(d.max()), float(d.mean())

def entry_point_xy(kps):
    """
    依据你画图逻辑：入口边是点0->点3，取该边的1/4处点作为入口点代表。
    返回 (2,) float
    """
    p0 = np.asarray(kps[0], float)
    p3 = np.asarray(kps[3], float)
    return p0 + 0.25 * (p3 - p0)

def entry_consistent(pred_kps, gt_kps, thresh=0.1):
    """
    入口点一致性：入口代表点欧式距离 <= thresh (米)
    """
    ep = entry_point_xy(pred_kps)
    eg = entry_point_xy(gt_kps)
    return np.linalg.norm(ep - eg) <= float(thresh)

def filter_by_choice(items, cat_choice=None, stat_choice=None):
    """
    对pred或gt列表做筛选：
    cat_choice: None或{int,...}  (0:Perp,1:Para,2:Others)
    stat_choice: None或{int,...}
    """
    if cat_choice is None and stat_choice is None:
        return items
    out = []
    for it in items:
        cond_cat = (cat_choice is None) or (it["cat"] in cat_choice)
        cond_stat = (stat_choice is None) or (it["stat"] in stat_choice)
        if cond_cat and cond_stat:
            out.append(it)
    return out

# ========== 匹配与度量累计 ==========
def match_slots(preds, gts,
                kp_max_thresh=0.05, iou_thresh=0.8, entry_thresh=0.1):
    """
    基于“严格TP规则”做一对一匹配：
      1) 四点对应最大距离 <= kp_max_thresh
      2) IoU >= iou_thresh
      3) 入口点一致（<= entry_thresh）
    返回:
      matches: list of (pi, gi, iou)
      pred_unmatched: set
      gt_unmatched: set
    """
    if len(preds) == 0 or len(gts) == 0:
        return [], set(range(len(preds))), set(range(len(gts)))

    # 预计算所有可行对
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = polygon_iou_quad(p["kps"], g["kps"])
            kp_max, kp_mean = kp_dist_metrics(p["kps"], g["kps"])
            ent_ok = entry_consistent(p["kps"], g["kps"], entry_thresh)
            ok = (kp_max <= kp_max_thresh) and (iou >= iou_thresh) and ent_ok
            if ok:
                # 排序优先按 IoU 高
                pairs.append((pi, gi, iou))

    # 贪心按IoU从高到低选不冲突的配对
    pairs.sort(key=lambda x: -x[2])
    matched_p, matched_g = set(), set()
    matches = []
    for pi, gi, iou in pairs:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        matches.append((pi, gi, iou))

    pred_unmatched = set(range(len(preds))) - matched_p
    gt_unmatched = set(range(len(gts))) - matched_g
    return matches, pred_unmatched, gt_unmatched

class MetricCounter:
    def __init__(self):
        self.det_tp = 0
        self.det_fp = 0
        self.det_fn = 0

        # 类型与状态的混淆统计（仅对已匹配样本评判对错）
        self.type_tp = 0
        self.type_fp = 0
        self.type_fn = 0

        # status分三类分别统计
        self.status_tp = defaultdict(int)  # key: class id
        self.status_fp = defaultdict(int)
        self.status_fn = defaultdict(int)

        # 计数筛选后的基数，便于精确denominator
        self.pred_count = 0
        self.gt_count = 0

    @staticmethod
    def _prf1(tp, fp, fn, eps=1e-9):
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        return prec, rec, f1

    def report(self):
        det_p, det_r, det_f1 = self._prf1(self.det_tp, self.det_fp, self.det_fn)
        type_p, type_r, type_f1 = self._prf1(self.type_tp, self.type_fp, self.type_fn)

        stat_metrics = {}
        for cid in [0,1,2]:
            p, r, f1 = self._prf1(self.status_tp[cid], self.status_fp[cid], self.status_fn[cid])
            stat_metrics[STAT_ID2NAME[cid]] = {"precision": p, "recall": r, "f1": f1}

        return {
            "DETECTION(strict-3-conds)": {"precision": det_p, "recall": det_r, "f1": det_f1,
                                          "tp": self.det_tp, "fp": self.det_fp, "fn": self.det_fn,
                                          "pred": self.pred_count, "gt": self.gt_count},
            "TYPE(cls@matched)": {"precision": type_p, "recall": type_r, "f1": type_f1,
                                  "tp": self.type_tp, "fp": self.type_fp, "fn": self.type_fn},
            "STATUS-3way(cls@matched)": stat_metrics
        }

def update_metrics(counter: MetricCounter,
                   preds, gts,
                   kp_max_thresh=0.05, iou_thresh=0.8, entry_thresh=0.1,
                   cat_choice=None, stat_choice=None):
    """
    对单帧进行累计：
      preds: list[dict] —— 来自 decode_pl_pred(pl_pred[0])
      gts:   list[dict] —— 来自 decode_pl_gt(all_data['curr']['parking_lots'])
      cat_choice: None 或 {0},{1},{2} 例如只评估 Para: {1}
      stat_choice: None 或 {0},{1},{2}
    """
    preds_f = filter_by_choice(preds, cat_choice, stat_choice)
    gts_f   = filter_by_choice(gts,   cat_choice, stat_choice)
    counter.pred_count += len(preds_f)
    counter.gt_count   += len(gts_f)

    # 匹配（严格TP规则）
    matches, pred_unmatched, gt_unmatched = match_slots(
        preds_f, gts_f, kp_max_thresh, iou_thresh, entry_thresh
    )
    # 检测级别统计
    counter.det_tp += len(matches)
    counter.det_fp += len(pred_unmatched)
    counter.det_fn += len(gt_unmatched)

    # 类型与状态分类统计：仅在已匹配的对上评估
    for (pi, gi, _) in matches:
        p = preds_f[pi]; g = gts_f[gi]
        # 类型
        if p["cat"] == g["cat"]:
            counter.type_tp += 1
        else:
            counter.type_fp += 1   # 预测为该类别的错误（等价：匹配上了但类型不一致）
            counter.type_fn += 1   # 同时对GT来说该类别也未被正确预测
        # 状态（逐类） —— 常见做法：对每一类都在匹配对上判断是否命中该类
        if p["stat"] == g["stat"]:
            counter.status_tp[g["stat"]] += 1
        else:
            counter.status_fp[p["stat"]] += 1
            counter.status_fn[g["stat"]] += 1

# ========== 与你的 offline infer 循环对接示例 ==========
"""
把下面这段示例嵌入你的for循环，注意：
- 从模型输出 pl_pred 拿到单张 pl_pred_item = pl_pred[0]，再 decode_pl_pred()
- 从 GT all_data['curr']['parking_lots'] 走 decode_pl_gt()

示例（伪代码嵌入点）：

counter = MetricCounter()

for idx in range(start_idx, end_idx):
    ...
    with torch.no_grad():
        infer_data = preprocess(args.config, all_data, input_params)
        pl_pred = model(return_loss=False, rescale=True, **infer_data)
        ...
        # ===== 解码预测与GT =====
        pred_list = decode_pl_pred(pl_pred[0])
        gt_list   = decode_pl_gt(all_data['curr']['parking_lots'])

        # ===== 无筛选（总体评估）=====
        update_metrics(counter, pred_list, gt_list,
                       kp_max_thresh=0.05, iou_thresh=0.8, entry_thresh=0.1,
                       cat_choice=None, stat_choice=None)

        # ===== 示例：只评估 Para（平行/水平）=====
        # update_metrics(counter, pred_list, gt_list, cat_choice={1})

        # ===== 示例：只评估 Vac（空闲）=====
        # update_metrics(counter, pred_list, gt_list, stat_choice={0})

# 循环结束后：
print(counter.report())
"""

# ========== 可选：简单的打印函数 ==========
def pretty_print_report(report_dict):
    det = report_dict["DETECTION(strict-3-conds)"]
    print("\n=== Detection (strict) ===")
    print(f"Precision: {det['precision']:.4f}  Recall: {det['recall']:.4f}  F1: {det['f1']:.4f}")
    print(f"TP={det['tp']}  FP={det['fp']}  FN={det['fn']}  (pred={det['pred']}, gt={det['gt']})")

    typ = report_dict["TYPE(cls@matched)"]
    print("\n=== Type (on matched) ===")
    print(f"Precision: {typ['precision']:.4f}  Recall: {typ['recall']:.4f}  F1: {typ['f1']:.4f}")

    print("\n=== Status (3-way, on matched) ===")
    for name, m in report_dict["STATUS-3way(cls@matched)"].items():
        print(f"{name:>8s} -> P: {m['precision']:.4f}  R: {m['recall']:.4f}  F1: {m['f1']:.4f}")
