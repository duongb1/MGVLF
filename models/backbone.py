# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d “đóng băng”: thống kê & tham số affine cố định.
    (Bản từ torchvision, thêm eps trước rsqrt để tránh NaN.)
    """
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        # ---- FREEZE RULE: chỉ layer2/3/4 trainable khi train_backbone=True ----
        trainable_layers = {'layer2', 'layer3', 'layer4'} if train_backbone else set()
        for name, parameter in backbone.named_parameters():
            allow_train = any(t in name for t in trainable_layers)
            parameter.requires_grad_(bool(allow_train))

        if return_interm_layers:
            # Multi-scale: C3/C4/C5
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.num_channels = [512, 1024, 2048]
        else:
            # Single-scale: C5
            return_layers = {"layer4": "0"}
            self.num_channels = [2048]

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)  # OrderedDict[str, Tensor]
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None
        for name, x in xs.items():
            # mask: True=padding; nội suy nearest để giữ nhị phân
            mask = F.interpolate(m[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with FrozenBatchNorm (có tùy chọn ImageNet-pretrained)."""
    def __init__(self, name: str, train_backbone: bool,
                 return_interm_layers: bool, dilation: bool,
                 pretrained: bool = True):
        ctor = getattr(torchvision.models, name)

        try:
            # API torchvision mới: dùng 'weights'
            if pretrained and name == "resnet50":
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            backbone = ctor(
                weights=weights,
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=FrozenBatchNorm2d,
            )
        except TypeError:
            # API cũ: dùng 'pretrained'
            backbone = ctor(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=bool(pretrained),
                norm_layer=FrozenBatchNorm2d,
            )

        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # Dict[str, NestedTensor]
        out: List[NestedTensor] = []
        pos = []
        # Duyệt theo thứ tự key để ổn định (layer1->4)
        for key in sorted(xs.keys(), key=lambda k: int(k)):
            x = xs[key]
            out.append(x)
            # position encoding ảnh (2D), dtype khớp feature map
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    # hidden_dim mặc định 256
    hidden_dim = int(getattr(args, 'hidden_dim', 256))
    if not hasattr(args, 'num_pos_feats'):
        args.num_pos_feats = hidden_dim // 2

    # Positional encoding (2D)
    position_embedding = build_position_encoding(
        args, position_embedding=getattr(args, 'img_pe_type', 'sine')
    )

    # Train backbone nếu lr_backbone > 0
    train_backbone = getattr(args, 'lr_backbone', 0.0) > 0.0

    # Multi-scale khi masks=True hoặc aux_loss=True
    return_interm_layers = bool(
        getattr(args, "masks", False) or getattr(args, "aux_loss", False)
    )

    backbone = Backbone(
        name=args.backbone,
        train_backbone=train_backbone,
        return_interm_layers=return_interm_layers,
        dilation=bool(getattr(args, "dilation", False)),
        pretrained=True,  # KHỞI TẠO từ ImageNet
    )

    # Joiner của bạn đã có/hoặc bạn đã thêm input_proj -> hidden_dim
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
