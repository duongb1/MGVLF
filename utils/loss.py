# utils/loss.py
import torch
import torch.nn.functional as F

_EPS = 1e-7

def _sanitize_xyxy(box: torch.Tensor):
    x1 = torch.minimum(box[..., 0], box[..., 2])
    y1 = torch.minimum(box[..., 1], box[..., 3])
    x2 = torch.maximum(box[..., 0], box[..., 2])
    y2 = torch.maximum(box[..., 1], box[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)

def Reg_Loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, reduction: str = "mean"):
    # pred_xyxy, gt_xyxy đều là xyxy_norm ∈ [0,1]
    pred = _sanitize_xyxy(pred_xyxy)
    gt   = _sanitize_xyxy(gt_xyxy)
    return F.smooth_l1_loss(pred, gt, reduction=reduction)

def GIoU_Loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, reduction: str = "mean"):
    # pred_xyxy, gt_xyxy đều là xyxy_norm ∈ [0,1]
    p = _sanitize_xyxy(pred_xyxy)
    g = _sanitize_xyxy(gt_xyxy)

    lt = torch.maximum(p[:, :2], g[:, :2])
    rb = torch.minimum(p[:, 2:], g[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_g = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
    union  = area_p + area_g - inter + _EPS
    iou    = inter / union

    enc_lt = torch.minimum(p[:, :2], g[:, :2])
    enc_rb = torch.maximum(p[:, 2:], g[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1] + _EPS

    giou = iou - (enc_area - union) / enc_area
    loss = 1.0 - giou
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

