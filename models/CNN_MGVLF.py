# CNN_MGVLF.py
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_vis_transformer, build_transformer, build_de
from models.position_encoding import build_position_encoding


# ===== 1D learned positional encoding for fusion sequence =====
class Learned1DPos(nn.Module):
    """
    Return a (B, S, C) positional tensor for a sequence of length S.
    Uses nn.Embedding with a large max_len; safe for variable S at runtime.
    """
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x_bs_c: torch.Tensor) -> torch.Tensor:
        """
        x_bs_c: (B, S, C)  -> returns (B, S, C)
        """
        B, S, C = x_bs_c.shape
        if S > self.max_len:
            raise RuntimeError(f"S={S} exceeds max_len={self.max_len} of Learned1DPos")
        idx = torch.arange(S, device=x_bs_c.device)  # (S,)
        pos = self.pe(idx).unsqueeze(0).expand(B, S, -1)  # (B,S,C)
        return pos


class CNN_MGVLF(nn.Module):
    """ This is the MLCM module """
    def __init__(self, backbone, transformer, DE, position_encoding, max_query_len: int = 40):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer      # EN in your notation
        self.DE = DE                        # decoder/encoder block before EN
        hidden_dim = transformer.d_model
        self.pos = position_encoding        # 2D pos for image features

        # dùng tham số max_query_len rõ ràng (không dùng args.time)
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

        # Sau khi self.backbone = backbone
        hidden_dim = transformer.d_model
        num_chs = getattr(backbone, "num_channels", [2048])
        if not isinstance(num_chs, (list, tuple)):
            num_chs = [num_chs]

        if len(num_chs) == 1:
            # single-scale (C5)
            self.input_proj = nn.Conv2d(num_chs[0], hidden_dim, kernel_size=1)
        else:
            # multi-scale (C3/C4/C5)
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ) for c in num_chs
            ])
        self.l_proj = torch.nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(), )

    def get_mask(self, nextFeatureMap, beforeMask):
        x = nextFeatureMap
        m = beforeMask
        assert m is not None
        # dùng nearest để giữ nhị phân
        mask = F.interpolate(m[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]
        return mask

    def forward(self, img, mask, word_mask, wordFeature, sentenceFeature):
        """
        img: (B,3,H,W), mask: (B,H,W) True=pad
        word_mask: (B,L) with 1=real, 0=pad (HF convention) -> convert to True=pad where needed
        wordFeature: (B,L,768), sentenceFeature: (B,768)
        """
        samples = NestedTensor(img, mask)
        outs = self.backbone(samples)
        
        # ---- unpack backbone outputs to list of Tensor ----
        if isinstance(outs, tuple):
            feats_any = outs[0]  # (feats, pos) or (feats, masks, pos)
            pos = outs[1] if len(outs) > 1 else None
        else:
            feats_any = outs
            pos = None

        # raw_feats keep original channels: [512, 1024, 2048] = (C3, C4, C5)
        raw_feats = [f.tensors if hasattr(f, "tensors") else f for f in feats_any]
        # ensure order C3,C4,C5
        assert len(raw_feats) >= 3, f"expect at least 3 feature maps, got {len(raw_feats)}"
        c3, c4, c5 = raw_feats[-3], raw_feats[-2], raw_feats[-1]

        # proj_feats map each to hidden_dim=256 for transformer/fusion
        if isinstance(self.input_proj, nn.ModuleList):
            proj_feats = [proj(f) for proj, f in zip(self.input_proj, [c3, c4, c5])]
        else:
            # single-scale fallback: only C5
            proj_feats = [self.input_proj(c5)]

        # Get masks from original features
        mask4 = feats_any[-1].mask if hasattr(feats_any[-1], "mask") else mask
        bs, _, h, w = c5.shape

        # pyramid convs - CNN branch must take raw C5 (2048 ch)
        conv6_1 = self.conv6_1(c5)
        conv6_2 = self.conv6_2(conv6_1)
        conv7_1 = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(conv7_1)
        conv8_1 = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(conv8_1)

        # Use projected features for transformer/fusion (256 ch)
        v_single = proj_feats[-1]  # P5 @ 256 ch
        
        # flatten visual feats
        conv5 = v_single  # Use projected feature for transformer
        fv1 = conv5.view(bs, conv5.shape[1], -1)  # Use actual channel count
        fv2 = conv6_2.view(bs, 256, -1)
        fv3 = conv7_2.view(bs, 256, -1)
        fv4 = conv8_2.view(bs, 256, -1)

        # masks for each level
        fv2_mask = self.get_mask(conv6_2, mask4)
        fv3_mask = self.get_mask(conv7_2, fv2_mask)
        fv4_mask = self.get_mask(conv8_2, fv3_mask)

        # 2D positional encodings for each level (match dtype with tensors)
        # Get pos from backbone output properly
        if pos is None and len(outs) > 1:
            pos = outs[1]
        
        if pos is not None:
            pos1 = pos[-1] if isinstance(pos, list) else pos
        else:
            # Generate pos encoding if not available
            pos1 = self.pos(NestedTensor(v_single, mask4)).to(v_single.dtype)
        
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)

        fvpos1 = pos1.view(bs, 256, -1)
        fvpos2 = pos2.view(bs, 256, -1)
        fvpos3 = pos3.view(bs, 256, -1)
        fvpos4 = pos4.view(bs, 256, -1)

        # concat multi-scale visual tokens (B,256,Lv) -> (Lv,B,256)
        fv = torch.cat([fv1, fv2, fv3, fv4], dim=2).permute(2, 0, 1)
        fvpos = torch.cat([fvpos1, fvpos2, fvpos3, fvpos4], dim=2).permute(2, 0, 1)

        # text tokens: concat words + sentence as last word
        textFeature = torch.cat([wordFeature, sentenceFeature.unsqueeze(1)], dim=1)  # (B,L+1,768)
        fl = self.l_proj(textFeature)                                                # (B,L+1,256)
        fl = fl.permute(1, 0, 2)                                                     # (L+1,B,256)

        # build masks
        # HF: 1=real, 0=pad  -> True=pad
        word_pad = (word_mask == 0)                              # (B,L) True=pad
        sentence_pad = torch.zeros((bs, 1), dtype=torch.bool, device=word_mask.device)
        text_pad = torch.cat([word_pad, sentence_pad], dim=1)    # (B,L+1)

        vis_pad = torch.cat([
            mask4.view(bs, -1),
            fv2_mask.view(bs, -1),
            fv3_mask.view(bs, -1),
            fv4_mask.view(bs, -1),
        ], dim=1).to(torch.bool)                                 # (B,Lv)

        # pos for text (use embedding slice up to Nt+1)
        Nt = wordFeature.size(1)   # number of word tokens before adding sentence token
        flpos = self.text_pos_embed.weight[:Nt+1].unsqueeze(1).repeat(1, bs, 1)  # (Nt+1,B,256)

        # concat pos and tokens for DE stage
        fvl = torch.cat((fv, fl), dim=0)                 # (Lv+L+1, B, 256)
        fvlpos = torch.cat((fvpos, flpos), dim=0)        # (Lv+L+1, B, 256)
        fvl_mask = torch.cat((vis_pad, text_pad), dim=1) # (B, Lv+L+1), True=pad

        # run DE
        out_layers = self.DE(
            fv1.permute(2, 0, 1),    # (Lv1, B, 256) as query (nếu DE của bạn mong đợi theo design cũ)
            fvl,                     # memory
            fvl_mask,                # key_padding_mask (True=pad)
            fvlpos,                  # memory pos
            fvpos1.permute(2, 0, 1)  # query pos (for fv1)
        )
        fv1_encode = out_layers[-1].permute(1, 2, 0)     # (B,256,H1*W1)

        refineFeature = fv1_encode.view(bs, 256, h, w)
        out = self.transformer(refineFeature, mask4, pos1)
        return out


class VLFusion(nn.Module):
    def __init__(self, transformer, pos_1d: nn.Module):
        """ Fusion block with learnable [pr] token at the BEGINNING of the sequence. """
        super().__init__()
        self.transformer = transformer
        self.pos_1d = pos_1d
        hidden_dim = transformer.d_model
        self.pr = nn.Embedding(1, hidden_dim)

        self.v_proj = torch.nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.l_proj = torch.nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
        )

    def forward(self, fv, fl, word_mask=None):
        """
        fv: (B,256,H, W)
        fl: (B,L,768) word-level from BERT (no sentence token here)
        word_mask: (B,L) 1=real,0=pad (HF)  -> convert to True=pad here
        returns last layer output (B,256,S) with token 0 = [pr]
        """
        B, C, H, W = fv.shape
        _, L, _    = fl.shape

        # (1) Visual tokens: (B,256,Lv)
        pv = fv.view(B, C, -1)                                     # (B,256,Lv)
        pv = self.v_proj(pv.transpose(1, 2)).transpose(1, 2)       # (B,256,Lv)

        # (2) Word tokens -> 256: (B,256,L)
        pl = self.l_proj(fl).transpose(1, 2)                       # (B,256,L)

        # (3) Learnable pr token at BEGINNING: (B,256,1)
        pr = self.pr.weight.unsqueeze(0).expand(B, -1, -1)         # (B,1,256)
        pr = pr.transpose(1, 2).contiguous()                       # (B,256,1)

        # (4) Sequence: [pr | pv | pl] -> (B,256,S)
        x0 = torch.cat((pr, pv, pl), dim=2)                        # S = 1 + Lv + L

        # (5) 1D PE on sequence (B,S,256)
        pos = self.pos_1d(x0.transpose(1, 2)).to(x0.dtype)         # (B,S,256)

        # (6) key_padding_mask (B,S) True=pad
        Lv = pv.shape[2]
        if word_mask is not None:
            text_pad = (word_mask == 0).to(x0.device)              # (B,L) True=pad
        else:
            text_pad = torch.zeros((B, L), dtype=torch.bool, device=x0.device)
        pr_pad  = torch.zeros((B, 1),  dtype=torch.bool, device=x0.device)
        vis_pad = torch.zeros((B, Lv), dtype=torch.bool, device=x0.device)
        mask = torch.cat([pr_pad, vis_pad, text_pad], dim=1)       # (B,S)

        # (7) Transformer fusion: expects src=(B,C,S), pos=(B,S,C), mask=(B,S)
        out = self.transformer(x0, mask, pos)                      # list/layers
        return out[-1]                                             # (B,256,S)


def build_CNN_MGVLF(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)  # ResNet-50
    EN = build_vis_transformer(args) # visual encoder
    DE = build_de(args)
    pos = build_position_encoding(args, position_embedding=getattr(args, 'img_pe_type', 'sine'))

    model = CNN_MGVLF(
        backbone=backbone,
        transformer=EN,
        DE=DE,
        position_encoding=pos,
        max_query_len=getattr(args, "max_query_len", 40),  # dùng --max_query_len
    )
    return model


def build_VLFusion(args):
    transformer = build_transformer(args)
    # dùng PE 1D học được ngay tại đây (không phụ thuộc build_position_encoding)
    pos_1d = Learned1DPos(d_model=transformer.d_model, max_len=getattr(args, "max_fusion_len", 8192))
    model = VLFusion(transformer, pos_1d)
    return model
