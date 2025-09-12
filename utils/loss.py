# utils/loss.py
import torch
import torch.nn.functional as F

_EPS = 1e-7

def _sanitize_xyxy(box: torch.Tensor):
    """
    box: (..., 4) in xyxy (normalized [0,1] or pixels, miễn cùng hệ)
    đảm bảo x1<=x2, y1<=y2 và clamp vào [0,1] nếu có thể suy ra miền.
    """
    x1 = torch.minimum(box[..., 0], box[..., 2])
    y1 = torch.minimum(box[..., 1], box[..., 3])
    x2 = torch.maximum(box[..., 0], box[..., 2])
    y2 = torch.maximum(box[..., 1], box[..., 3])
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    return out

def Reg_Loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, reduction: str = "mean"):
    """
    SmoothL1 trên xyxy. Cả pred và gt nên cùng hệ (khuyến nghị [0,1]).
    """
    pred = _sanitize_xyxy(pred_xyxy)
    gt   = _sanitize_xyxy(gt_xyxy)
    loss = F.smooth_l1_loss(pred, gt, reduction=reduction)
    return loss

def GIoU_Loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, reduction: str = "mean"):
    """
    Generalized IoU loss: 1 - GIoU.
    pred_xyxy, gt_xyxy: (B,4) cùng hệ toạ độ (khuyến nghị [0,1]).
    """
    p = _sanitize_xyxy(pred_xyxy)
    g = _sanitize_xyxy(gt_xyxy)

    # intersection
    lt = torch.maximum(p[:, :2], g[:, :2])                 # (B,2)
    rb = torch.minimum(p[:, 2:], g[:, 2:])                 # (B,2)
    wh = (rb - lt).clamp(min=0)                            # (B,2)
    inter = wh[:, 0] * wh[:, 1]                            # (B,)

    # areas & union
    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_g = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
    union  = area_p + area_g - inter + _EPS
    iou    = inter / union

    # smallest enclosing box
    enc_lt = torch.minimum(p[:, :2], g[:, :2])
    enc_rb = torch.maximum(p[:, 2:], g[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1] + _EPS

    giou = iou - (enc_area - union) / enc_area
    loss = 1.0 - giou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss  # none

