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


def bbox_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    IoU giữa 2 tập bbox numpy.
    box1: (N,4), box2: (M,4), định dạng xyxy.
    return: (N, M) IoU.
    """
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(box1[:, None, 2], box2[None, :, 2]) - np.maximum(box1[:, None, 0], box2[None, :, 0])
    ih = np.minimum(box1[:, None, 3], box2[None, :, 3]) - np.maximum(box1[:, None, 1], box2[None, :, 1])

    iw = np.clip(iw, 0, None)
    ih = np.clip(ih, 0, None)
    inter = iw * ih

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    ua = area1[:, None] + area2[None, :] - inter
    ua = np.maximum(ua, np.finfo(float).eps)

    return inter / ua


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, x1y1x2y2: bool = True):
    """
    PyTorch IoU cho hai tập bbox.
    - Nếu x1y1x2y2=True, cả 2 là xyxy; ngược lại, là cxcywh.
    Trả về: (iou, inter_area, union_area) đều là Tensor dạng (N,).
    """
    eps = 1e-16
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] * 0.5, box1[:, 0] + box1[:, 2] * 0.5
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] * 0.5, box1[:, 1] + box1[:, 3] * 0.5
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] * 0.5, box2[:, 0] + box2[:, 2] * 0.5
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] * 0.5, box2[:, 1] + box2[:, 3] * 0.5

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0)
    inter_h = torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    b2_area = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union_area = b1_area + b2_area - inter_area + eps

    iou = inter_area / union_area
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
    LR decay sau epoch 60 (giống code cũ).
    Tương thích 2 nhóm (rest, visu) hoặc 3 nhóm (rest, visu, text).
    """
    lr = args.lr * args.lr_dec if epoch >= 60 else args.lr
    # group 0: rest
    optimizer.param_groups[0]['lr'] = lr
    # group 1: visu
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr / 10.0
    # group 2: text (nếu có)
    if len(optimizer.param_groups) > 2:
        optimizer.param_groups[2]['lr'] = lr / 10.0
