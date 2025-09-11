#CNN_MGVLF.py
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_vis_transformer, build_transformer,build_de
from models.position_encoding import build_position_encoding


class CNN_MGVLF(nn.Module):
    """ This is the MLCM module """
    def __init__(self, backbone, transformer, DE, position_encoding, max_query_len: int = 40):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.DE = DE
        hidden_dim = transformer.d_model
        self.pos = position_encoding

        # không hard-code 41 nữa
        self.max_query_len = int(max_query_len)
        self.text_pos_embed = nn.Embedding(self.max_query_len + 1, hidden_dim)

        self.conv6_1 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.l_proj = torch.nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(), )

    def get_mask(self, nextFeatureMap, beforeMask):
        x = nextFeatureMap
        m = beforeMask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        return mask

    def forward(self, img, mask, word_mask, wordFeature, sentenceFeature):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = NestedTensor(img, mask)
        features, pos = self.backbone(samples)
        featureMap4, mask4 = features[3].decompose()
        bs, c, h, w = featureMap4.shape

        conv6_1 = self.conv6_1(featureMap4)
        conv6_2 = self.conv6_2(conv6_1)
        conv7_1 = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(conv7_1)
        conv8_1 = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(conv8_1)

        conv5 = self.input_proj(featureMap4)
        fv1 = conv5.view(bs, 256, -1)
        fv2 = conv6_2.view(bs, 256, -1)
        fv3 = conv7_2.view(bs, 256, -1)
        fv4 = conv8_2.view(bs, 256, -1)
        fv2_mask = self.get_mask(conv6_2, mask4)
        fv3_mask = self.get_mask(conv7_2, fv2_mask)
        fv4_mask = self.get_mask(conv8_2, fv3_mask)

        pos1 = pos[-1]
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)
        fvpos1 = pos1.view(bs, 256, -1)
        fvpos2 = pos2.view(bs, 256, -1)
        fvpos3 = pos3.view(bs, 256, -1)
        fvpos4 = pos4.view(bs, 256, -1)

        fv = torch.cat((fv1, fv2), dim=2)
        fv = torch.cat((fv, fv3), dim=2)
        fv = torch.cat((fv, fv4), dim=2)
        fv = fv.permute(2, 0, 1)
        textFeature = torch.cat([wordFeature, sentenceFeature.unsqueeze(1)], dim=1)
        fl = self.l_proj(textFeature)
        fl = fl.permute(1, 0, 2)
        fvl = torch.cat((fv, fl), dim=0)

        word_mask = word_mask.to(torch.bool)
        word_mask = ~word_mask
        sentence_mask = torch.zeros((bs, 1)).to(word_mask.device).to(torch.bool)
        text_mask = torch.cat((word_mask, sentence_mask), dim=1)
        vis_mask = torch.cat((mask4.view(bs, -1), fv2_mask.view(bs, -1)), dim=1)
        vis_mask = torch.cat((vis_mask, fv3_mask.view(bs, -1)), dim=1)
        vis_mask = torch.cat((vis_mask, fv4_mask.view(bs, -1)), dim=1)
        fvl_mask = torch.cat((vis_mask, text_mask), dim=1)

        # thay vì: flpos = self.text_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        Nt = wordFeature.size(1)  # số word tokens (thường = max_query_len sau pad)
        flpos = self.text_pos_embed.weight[:Nt+1].unsqueeze(1).repeat(1, bs, 1)  # (Nt+1, B, C)
        fvpos = torch.cat((fvpos1, fvpos2), dim=2)
        fvpos = torch.cat((fvpos, fvpos3), dim=2)
        fvpos = torch.cat((fvpos, fvpos4), dim=2)
        fvpos = fvpos.permute(2, 0, 1)
        fvlpos = torch.cat((fvpos, flpos), dim=0)

        out_layers = self.DE(fv1.permute(2, 0, 1), fvl, fvl_mask, fvlpos,fvpos1.permute(2, 0, 1))
        fv1_encode = out_layers[-1].permute(1, 2, 0)

        refineFeature = fv1_encode.view(bs, 256, h, w)
        out = self.transformer(refineFeature, mask4, pos1)
        return out


class VLFusion(nn.Module):
    def __init__(self, transformer, pos):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: no use
            """
        super().__init__()
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model
        self.pr = nn.Embedding(1, hidden_dim)

        self.v_proj = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),)
        self.l_proj = torch.nn.Sequential(
          nn.Linear(768, 256),
          nn.ReLU(),)

    def forward(self, fv, fl, word_mask=None):

        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        B, C, H, W = fv.shape
        _, L, _    = fl.shape

        # (1) Visual tokens: (B, 256, H*W)
        pv = fv.view(B, C, -1)                  # (B,256,Lv)
        pv = self.v_proj(pv.transpose(1, 2)).transpose(1, 2)  # Linear->(B,Lv,256)->(B,256,Lv)

        # (2) Word tokens: (B, 256, L)
        pl = self.l_proj(fl).transpose(1, 2)    # (B,256,L)

        # (3) Learnable pr token: (B,256,1)
        pr = self.pr.weight.unsqueeze(0).expand(B, -1).unsqueeze(2)  # (B,256,1)

        # (4) Chuỗi fusion: src=(B,256,S)
        x0 = torch.cat((pv, pl, pr), dim=2)     # (B,256,S) với S = Lv + L + 1

        # (5) Positional encoding 1D cho chuỗi: pos=(B,S,256)
        # self.pos là learned-1D PE (đã set ở build_VLFusion)
        pos = self.pos(x0.transpose(1, 2)).to(x0.dtype)  # (B,S,256)

        # (6) Padding mask: True=padding theo từng đoạn [vis | text | pr]
        Lv = pv.shape[2]
        if word_mask is not None:
            # word_mask: 1=real, 0=pad  ->  True=pad
            text_pad = (~word_mask.bool()).to(x0.device)   # (B,L)
        else:
            text_pad = torch.zeros((B, L), dtype=torch.bool, device=x0.device)

        vis_pad = torch.zeros((B, Lv), dtype=torch.bool, device=x0.device)
        pr_pad  = torch.zeros((B, 1),  dtype=torch.bool, device=x0.device)
        mask = torch.cat([vis_pad, text_pad, pr_pad], dim=1)   # (B,S), True=pad

        # (7) Transformer fusion: expects src=(B,C,S), pos=(B,S,C), mask=(B,S)
        out = self.transformer(x0, mask, pos)
        return out[-1]



def build_CNN_MGVLF(args):
    device = torch.device(args.device)
    backbone = build_backbone(args) # ResNet 50
    EN = build_vis_transformer(args)
    DE = build_de(args)
    pos = build_position_encoding(args, position_embedding='sine')

    model = CNN_MGVLF(
        backbone=backbone,
        transformer=EN,
        DE=DE,
        position_encoding=pos,
        max_query_len=getattr(args, "time", 40),  # dùng --time từ main.py
    )
    return model


def build_VLFusion(args):
    transformer = build_transformer(args)
    # DÙNG PE 1D cho chuỗi fusion, không dùng PE ảnh 2D:
    pos = build_position_encoding(args, position_embedding='learned1d')
    model = VLFusion(transformer, pos)
    return model

