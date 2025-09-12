# utils/utils.py
import numpy as np
import torch


class AverageMeter(object):
    """Compute and store average + running sum/count."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """[x1, y1, x2, y2] → [cx, cy, w, h] (giữ đơn vị gốc)."""
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) * 0.5
    y[:, 1] = (x[:, 1] + x[:, 3]) * 0.5
    y[:, 2] = (x[:, 2] - x[:, 0])
    y[:, 3] = (x[:, 3] - x[:, 1])
    return y


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2] (giữ đơn vị gốc)."""
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] * 0.5
    y[:, 1] = x[:, 1] - x[:, 3] * 0.5
    y[:, 2] = x[:, 0] + x[:, 2] * 0.5
    y[:, 3] = x[:, 1] + x[:, 3] * 0.5
    return y


# -------------------------
# BBox sanitation helpers
# -------------------------
def _sanitize_xyxy_t(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x1<=x2, y1<=y2 for torch tensors. x: (N,4)
    """
    x1 = torch.minimum(x[:, 0], x[:, 2])
    y1 = torch.minimum(x[:, 1], x[:, 3])
    x2 = torch.maximum(x[:, 0], x[:, 2])
    y2 = torch.maximum(x[:, 1], x[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1)


def _sanitize_xyxy_np(x: np.ndarray) -> np.ndarray:
    """
    Ensure x1<=x2, y1<=y2 for numpy arrays. x: (N,4)
    """
    x1 = np.minimum(x[:, 0], x[:, 2])
    y1 = np.minimum(x[:, 1], x[:, 3])
    x2 = np.maximum(x[:, 0], x[:, 2])
    y2 = np.maximum(x[:, 1], x[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1)


def bbox_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    IoU giữa 2 tập bbox numpy.
    box1: (N,4), box2: (M,4), định dạng xyxy.
    return: (N, M) IoU.
    """
    eps = np.finfo(np.float64).eps
    box1 = _sanitize_xyxy_np(box1.astype(np.float64, copy=False))
    box2 = _sanitize_xyxy_np(box2.astype(np.float64, copy=False))

    iw = np.clip(np.minimum(box1[:, None, 2], box2[None, :, 2]) - np.maximum(box1[:, None, 0], box2[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(box1[:, None, 3], box2[None, :, 3]) - np.maximum(box1[:, None, 1], box2[None, :, 1]), 0, None)
    inter = iw * ih

    area1 = np.clip(box1[:, 2] - box1[:, 0], 0, None) * np.clip(box1[:, 3] - box1[:, 1], 0, None)
    area2 = np.clip(box2[:, 2] - box2[:, 0], 0, None) * np.clip(box2[:, 3] - box2[:, 1], 0, None)

    ua = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(ua, eps)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, x1y1x2y2: bool = True):
    """
    PyTorch IoU cho hai tập bbox.
    - Nếu x1y1x2y2=True, cả 2 là xyxy; ngược lại, là cxcywh.
    Trả về: (iou, inter_area, union_area) đều là Tensor dạng (N,).
    """
    eps = 1e-7
    if not x1y1x2y2:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    box1 = _sanitize_xyxy_t(box1)
    box2 = _sanitize_xyxy_t(box2)

    inter_x1 = torch.maximum(box1[:, 0], box2[:, 0])
    inter_y1 = torch.maximum(box1[:, 1], box2[:, 1])
    inter_x2 = torch.minimum(box1[:, 2], box2[:, 2])
    inter_y2 = torch.minimum(box1[:, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    a1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    a2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union_area = a1 + a2 - inter_area
    iou = inter_area / union_area.clamp_min(eps)
    return iou, inter_area, union_area


def multiclass_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    Tính precision / recall / F1 cho mảng multi-label nhị phân (0/1 hoặc logits).
    pred, gt: shape (N, C), giá trị >0.5 xem như 1.
    """
    eps = 1e-6
    overall = {'precision': -1.0, 'recall': -1.0, 'f1': -1.0}
    NP, NR, NC = 0, 0, 0

    for ii in range(pred.shape[0]):
        pred_ind = (pred[ii] > 0.5).astype(int)
        gt_ind = (gt[ii] > 0.5).astype(int)
        inter = pred_ind * gt_ind

        NC += int(inter.sum())
        NP += int(pred_ind.sum())
        NR += int(gt_ind.sum())

    if NP > 0:
        overall['precision'] = float(NC) / NP
    if NR > 0:
        overall['recall'] = float(NC) / NR
    if NP > 0 and NR > 0:
        p = overall['precision']
        r = overall['recall']
        overall['f1'] = 2 * p * r / (p + r + eps)
    return overall


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Average Precision từ các điểm (recall, precision).
    Triển khai theo py-faster-rcnn.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]), axis=0)
    mpre = np.concatenate(([0.0], precision, [0.0]), axis=0)

    # precision envelope
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # các vị trí recall thay đổi
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def adjust_learning_rate(args, optimizer, epoch: int):
    """
    LR decay theo tham số, mặc định drop tại epoch 60.
    Tương thích 2 nhóm (rest, visu) hoặc 3 nhóm (rest, visu, text).
    """
    drop_at = getattr(args, "lr_drop", 60)
    dec = getattr(args, "lr_dec", 0.1)
    base_lr = getattr(args, "lr", 1e-4)
    lr = base_lr * dec if epoch >= drop_at else base_lr

    # group 0: rest
    optimizer.param_groups[0]['lr'] = lr
    # group 1: visu
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr / 10.0
    # group 2: text (nếu có)
    if len(optimizer.param_groups) > 2:
        optimizer.param_groups[2]['lr'] = lr / 10.0
