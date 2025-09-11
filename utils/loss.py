import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def Reg_Loss(output, target):
    sm_l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
    
    loss_x1 = sm_l1_loss(output[:,0], target[:,0])
    loss_x2 = sm_l1_loss(output[:,1], target[:,1])
    loss_y1 = sm_l1_loss(output[:,2], target[:,2])
    loss_y2 = sm_l1_loss(output[:,3], target[:,3])

    return (loss_x1+loss_x2+loss_y1+loss_y2)


def GIoU_Loss(pred_xyxy, gt_xyxy):
    """
    pred_xyxy, gt_xyxy: (B,4) ở pixel, dạng [x1,y1,x2,y2]
    Trả về: mean(1 - GIoU)
    """
    # Intersection
    max_xy = torch.min(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    min_xy = torch.max(pred_xyxy[:, :2], gt_xyxy[:, :2])
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter_area = inter[:, 0] * inter[:, 1]

    # Union
    area1 = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    union = area1 + area2 - inter_area + 1e-7
    iou = inter_area / union

    # Enclose
    enclose_lt = torch.min(pred_xyxy[:, :2], gt_xyxy[:, :2])
    enclose_rb = torch.max(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    enclose_wh = torch.clamp(enclose_rb - enclose_lt, min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + 1e-7

    giou = iou - (enclose_area - union) / enclose_area
    return (1.0 - giou).mean()