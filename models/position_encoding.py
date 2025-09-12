# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Sine/cosine 2D PE cho ảnh, giống 'Attention is All You Need' mở rộng cho hình ảnh.
    Trả về (B, C, H, W).
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned1D(nn.Module):
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


class PositionEmbeddingLearned2D(nn.Module):
    """
    Learned 2D positional embedding cho ẢNH. pos(x,y) = row_embed[y] + col_embed[x]
    Trả về (B, C, H, W).
    """
    def __init__(self, num_pos_feats=256, num_rows=256, num_cols=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_rows, num_pos_feats)
        self.col_embed = nn.Embedding(num_cols, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # (B, C, H, W)
        H, W = x.shape[-2], x.shape[-1]
        i = torch.arange(W, device=x.device).clamp(max=self.col_embed.num_embeddings - 1)
        j = torch.arange(H, device=x.device).clamp(max=self.row_embed.num_embeddings - 1)
        x_emb = self.col_embed(i)  # (W, C)
        y_emb = self.row_embed(j)  # (H, C)
        pos = y_emb[:, None, :] + x_emb[None, :, :]  # (H, W, C)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(x.size(0), -1, -1, -1)  # (B, C, H, W)


def build_position_encoding(args, position_embedding):
    if position_embedding in ('v2', 'sine'):
        N_steps = args.hidden_dim // 2
        return PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('learned2d', 'learned', 'v3'):
        return PositionEmbeddingLearned2D(
            num_pos_feats=args.hidden_dim,
            num_rows=getattr(args, 'pe_rows', 256),
            num_cols=getattr(args, 'pe_cols', 256),
        )
    elif position_embedding in ('learned1d', 'fusion1d'):
        return PositionEmbeddingLearned1D(
            d_model=args.hidden_dim,
            max_len=getattr(args, 'fusion_pe_max_len', 4096),
        )
    else:
        raise ValueError(f"not supported {position_embedding}")
