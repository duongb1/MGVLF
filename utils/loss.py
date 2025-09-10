# utils/loss.py
import torch
import torch.nn.functional as F

def bbox_xywh_to_xyxy(b):
    # b: (..., 4) in [0,1] (norm) hoặc pixel — dùng nhất quán với gt
    cx, cy, w, h = b[...,0], b[...,1], b[...,2], b[...,3]
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return torch.stack([x1,y1,x2,y2], dim=-1)

def iou_xyxy(a, b, eps=1e-7):
    # a,b: (B,4), tọa độ cùng hệ (pixel hoặc norm)
    x1 = torch.max(a[:,0], b[:,0])
    y1 = torch.max(a[:,1], b[:,1])
    x2 = torch.min(a[:,2], b[:,2])
    y2 = torch.min(a[:,3], b[:,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
    area_b = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
    union = area_a + area_b - inter + eps
    return (inter / union).unsqueeze(-1)  # (B,1)

def quality_loss(qhat, iou_t, weight=0.2):
    # qhat: (B,1) sigmoid; iou_t: (B,1) in [0,1]
    return F.mse_loss(qhat, iou_t) * weight
