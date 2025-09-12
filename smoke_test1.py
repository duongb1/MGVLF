# smoke_test.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === import project modules (điều chỉnh nếu khác đường dẫn) ===
from data_loader import RSVGDataset  # hoặc dataset.py tùy bạn
from models.model import MGVLF
from models.backbone import build_backbone  # dùng bên trong visumodel nếu cần
from utils.transforms import letterbox  # nếu muốn test riêng

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_path", type=str, required=True)
    ap.add_argument("--anno_path", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bert_model", type=str, default="bert-base-uncased")
    ap.add_argument("--aux_loss", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    ap.add_argument("--masks", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    return ap.parse_args()

def print_ok(name, extra=""):
    print(f"[OK] {name}{(' - ' + extra) if extra else ''}")

def print_warn(name, extra=""):
    print(f"[WARN] {name}{(' - ' + extra) if extra else ''}")

def check_env():
    print("=== ENV ===")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    print_ok("Environment visible")

def build_dataset(args):
    print("=== DATASET ===")
    ds = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        imsize=args.size,
        augment=False,
        split="train",
        testmode=False,
        splits_dir=args.splits_dir,
        bert_model=args.bert_model,
    )
    print_ok("RSVGDataset init", f"len={len(ds)}")
    return ds

import numpy as np
import torch

def collate_fn(batch):
    """
    Kỳ vọng mỗi item trong batch là:
      (img, pad_mask, word_id, word_mask, gt_box)
    - img: np.ndarray HxWxC (RGB/BGR) hoặc torch.Tensor CxHxW
    - pad_mask: np.ndarray HxW or torch.BoolTensor (True=pad)
    - word_id: np.ndarray/torch.LongTensor (L,)
    - word_mask: np.ndarray/torch.LongTensor (L,) with 1=real, 0=pad
    - gt_box: np.ndarray/torch.FloatTensor (4,) theo canvas sau letterbox [x1,y1,x2,y2] hoặc [cx,cy,w,h]
    """

    imgs, masks, word_ids, word_masks, gt_boxes = [], [], [], [], []

    def to_img_tensor(x):
        # -> torch.FloatTensor (C,H,W)
        if isinstance(x, torch.Tensor):
            t = x
            if t.ndim == 3 and t.shape[0] in (1,3):  # C,H,W
                return t.float()
            if t.ndim == 3 and t.shape[-1] in (1,3):  # H,W,C
                return t.permute(2,0,1).float()
            raise RuntimeError(f"Unexpected image tensor shape: {t.shape}")
        elif isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[-1] in (1,3):  # H,W,C
                t = torch.from_numpy(x).permute(2,0,1).contiguous().float()
                return t
            if x.ndim == 3 and x.shape[0] in (1,3):  # C,H,W
                return torch.from_numpy(x).contiguous().float()
            raise RuntimeError(f"Unexpected image array shape: {x.shape}")
        else:
            raise TypeError(f"img type must be tensor or ndarray, got {type(x)}")

    def to_mask_tensor(x):
        # -> torch.BoolTensor (H,W), True = pad
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.bool:
                x = x != 0
            return x.bool()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).bool()
        else:
            raise TypeError(f"mask type must be tensor or ndarray, got {type(x)}")

    def to_long_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.long()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).long()
        else:
            raise TypeError(f"long tensor expects ndarray/tensor, got {type(x)}")

    def to_float_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        else:
            raise TypeError(f"float tensor expects ndarray/tensor, got {type(x)}")

    for item in batch:
        img, pad_mask, wid, wmask, gt = item  # điều chỉnh nếu thứ tự khác
        imgs.append(to_img_tensor(img))
        masks.append(to_mask_tensor(pad_mask))
        word_ids.append(to_long_tensor(wid))
        word_masks.append(to_long_tensor(wmask))
        gt_boxes.append(to_float_tensor(gt))

    # Chuẩn hoá shape nhất quán
    # Ảnh: (B,C,H,W)
    imgs = torch.stack(imgs, dim=0)
    # Mask: (B,H,W)
    masks = torch.stack(masks, dim=0)
    # Text: (B,L)
    word_id = torch.stack(word_ids, dim=0)
    word_mask = torch.stack(word_masks, dim=0)
    # GT box: (B,4)
    gt_boxes = torch.stack(gt_boxes, dim=0)

    # Kiểm tra nhanh (1 lần)
    if not hasattr(collate_fn, "_once"):
        print("[COLLATE] imgs", imgs.shape, imgs.dtype)
        print("[COLLATE] masks", masks.shape, masks.dtype)
        print("[COLLATE] word_id", word_id.shape, word_id.dtype)
        print("[COLLATE] word_mask unique:", sorted(torch.unique(word_mask).tolist()))
        print("[COLLATE] gt_boxes", gt_boxes.shape, gt_boxes.dtype)
        collate_fn._once = True

    return imgs, masks, word_id, word_mask, gt_boxes


def check_polarity(word_mask):
    # HF: 1=real, 0=pad
    uniq = torch.unique(word_mask).tolist()
    assert all(x in [0,1] for x in uniq), f"word_mask must be 0/1 only, got {uniq}"
    key_pad = (word_mask == 0)  # True=pad
    assert key_pad.dtype == torch.bool, "key_pad must be bool"
    print_ok("Text mask polarity", f"unique={uniq}")
    return key_pad

def build_args_for_model(ns):
    # tạo một object đơn giản chứa các field mà model/backbone cần
    class A: pass
    a = A()
    a.size = ns.size
    a.images_path = ns.images_path
    a.anno_path = ns.anno_path
    a.splits_dir = ns.splits_dir
    a.lr = 1e-4
    a.lr_backbone = ns.lr_backbone
    a.batch_size = ns.batch_size
    a.resume = ""
    a.pretrain = ""  # bạn seed từ DETR trong model.__init__ rồi
    a.print_freq = 50
    a.savename = "smoke"
    a.seed = 13
    a.bert_model = ns.bert_model
    a.tunebert = True
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    a.masks = ns.masks
    a.aux_loss = ns.aux_loss
    a.backbone = "resnet50"
    a.dilation = False
    a.position_embedding = "sine"
    a.enc_layers = 6
    a.dec_layers = 6
    a.dim_feedforward = 2048
    a.hidden_dim = 256
    a.dropout = 0.1
    a.nheads = 8
    a.pre_norm = False
    a.img_pe_type = "sine"
    a.fusion_pe_max_len = 4096
    a.pe_rows = 256
    a.pe_cols = 256
    return a

@torch.no_grad()
def check_backbone_multiscale(model_visu, imgs, masks, args):
    try:
        from utils.misc import NestedTensor
        dev = next(model_visu.backbone.parameters()).device
        samples = NestedTensor(imgs.to(dev), masks.to(dev))

        outs = model_visu.backbone(samples)
        # 3 kiểu trả về phổ biến:
        # (feats, masks, pos)  or  (feats, pos)  or  ([NestedTensor...], pos)
        if isinstance(outs, tuple) and len(outs) == 3:
            feats, out_masks, pos = outs
        elif isinstance(outs, tuple) and len(outs) == 2:
            a, pos = outs
            # a có thể là list[Tensor] hoặc list[NestedTensor]
            if isinstance(a, (list, tuple)) and hasattr(a[0], "tensors"):
                feats = [nt.tensors for nt in a]
            else:
                feats = list(a)
        else:
            # Fallback: coi outs là list[NestedTensor] hoặc list[Tensor]
            feats = [x.tensors if hasattr(x, "tensors") else x for x in outs]

        n = len(feats)
        ch = [f.shape[1] for f in feats]
        print("[OK] Backbone forward")
        print("[DBG] maps:", n, "channels:", ch)

        expect = 3 if (args.aux_loss or args.masks) else 1
        if n != expect:
            print(f"[WARN] Unexpected #feature maps: got {n}, expect {expect} "
                  f"(aux_loss={args.aux_loss}, masks={args.masks})")
    except Exception as e:
        print("[WARN] Backbone debug skipped -", e)


def tiny_overfit_step(model, batch, device):
    imgs, masks, word_id, word_mask, gt_boxes = batch
    imgs, masks = imgs.to(device), masks.to(device)
    word_id, word_mask = word_id.to(device), word_mask.to(device)
    gt_boxes = gt_boxes.to(device)  # expected pixel x1,y1,x2,y2 on canvas

    # Forward: normalized cx,cy,w,h in [0,1]
    out = model(imgs, masks, word_id, word_mask)  # (B,4)

    # Convert pred to pixel x1,y1,x2,y2 on canvas
    B, C, H, W = imgs.shape
    cx = out[:, 0] * W
    cy = out[:, 1] * H
    ww = out[:, 2] * W
    hh = out[:, 3] * H
    x1 = cx - ww / 2
    y1 = cy - hh / 2
    x2 = cx + ww / 2
    y2 = cy + hh / 2
    pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # L1 sanity loss (just for smoke test)
    loss = (pred_boxes - gt_boxes).abs().mean()
    return pred_boxes, loss

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_env()

    ds = build_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_fn, drop_last=True)

    # Lấy 1 batch
    imgs, masks, word_id, word_mask, gt_boxes = next(iter(dl))
    print_ok("Batch fetched", f"shape={tuple(imgs.shape)}")

    # 1) Polarity
    key_pad = check_polarity(word_mask)

    # 2) Model build
    margs = build_args_for_model(args)
    model = MGVLF(bert_model=margs.bert_model, tunebert=True, args=margs).to(device)
    print("[DBG] input_proj type:", type(model.visumodel.input_proj).__name__)
    model.eval()
    print_ok("Model built")
    
    # Check if multi-scale is actually enabled
    print(f"[DBG] input_proj type: {type(model.visumodel.input_proj).__name__}")
    # Expect: 'ModuleList' when aux_loss=True (or masks=True); 'Conv2d' means single-scale.

    # 3) Backbone multi-scale check
    check_backbone_multiscale(model.visumodel, imgs, masks, margs)

    # 4) Forward pass
    imgs_d = imgs.to(device); masks_d = masks.to(device)
    word_id_d = word_id.to(device); word_mask_d = word_mask.to(device)
    out = model(imgs_d, masks_d, word_id_d, word_mask_d)  # (B,4) in [0,1]
    assert out.shape[-1] == 4, f"output must be (B,4), got {out.shape}"
    mn, mx = out.min().item(), out.max().item()
    print_ok("Forward pass", f"pred range=[{mn:.4f},{mx:.4f}]")

    # 5) Tiny overfit (1 step, kiểm loss giảm chạy được)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    batch = (imgs, masks, word_id, word_mask, gt_boxes)
    out0, loss0 = tiny_overfit_step(model, batch, device)
    opt.zero_grad(); loss0.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.1); opt.step()
    out1, loss1 = tiny_overfit_step(model, batch, device)
    print_ok("Tiny overfit step", f"loss0={loss0.item():.4f} -> loss1={loss1.item():.4f}")

    print("\n=== SMOKE TEST COMPLETED ===")

if __name__ == "__main__":
    main()
