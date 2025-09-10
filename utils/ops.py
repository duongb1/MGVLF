# utils/ops.py
import torch

def batched_index_select(tensor, indices):
    """
    tensor: (B, N, C)
    indices: (B, k) int64
    return: (B, k, C)
    """
    B, N, C = tensor.shape
    k = indices.shape[1]
    arange = torch.arange(B, device=tensor.device).unsqueeze(-1)  # (B,1)
    out = tensor[arange, indices, :]  # (B,k,C)
    return out
