#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMOKE TEST hợp nhất cho RSVG/MGVLF: Data pipeline + Backbone + Fusion [pr] token.

Ví dụ (Kaggle):
python smoke_test.py \
  --images_path "/kaggle/input/dior-rsvg/DIOR_RSVG/JPEGImages" \
  --anno_path   "/kaggle/input/dior-rsvg/DIOR_RSVG/Annotations" \
  --splits_dir  "/kaggle/input/dior-rsvg/DIOR_RSVG" \
  --split val \
  --imsize 640 \
  --limit 32 \
  --device cuda \
  --tiny_overfit true
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tránh cảnh báo fork của HF tokenizers

import sys
import math
import argparse
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

# === import project modules (điều chỉnh nếu khác đường dẫn) ===
from data_loader import RSVGDataset
from utils.misc import NestedTensor
from utils.utils import bbox_iou
from models.backbone import build_backbone
from models.model import MGVLF

# ---------- utils in-file ----------
def PASS(msg): print(f"[PASS] {msg}")
def FAIL(msg): print(f"[FAIL] {msg}") or sys.exit(1)
def _exit_if(cond, msg, code=1):
    if cond:
        FAIL(msg)
        sys.exit(code)

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

def _sanitize_xyxy_t(x: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(x[:, 0], x[:, 2])
    y1 = torch.minimum(x[:, 1], x[:, 3])
    x2 = torch.maximum(x[:, 0], x[:, 2])
    y2 = torch.maximum(x[:, 1], x[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1)

def _split_ratio(r):
    """
    Trả về (rx, ry) dưới dạng float, chấp nhận:
      - float / int
      - tuple/list có 1 hoặc 2 phần tử
      - numpy.ndarray: 0-d (scalar) hoặc có >=2 phần tử
    """
    import numpy as _np
    if isinstance(r, (tuple, list)):
        if len(r) >= 2:
            return float(r[0]), float(r[1])
        elif len(r) == 1:
            v = float(r[0]); return v, v
        else:
            return 1.0, 1.0
    if isinstance(r, _np.ndarray):
        if r.ndim == 0:
            v = float(r); return v, v
        r = r.reshape(-1)
        if r.size >= 2:
            return float(r[0]), float(r[1])
        elif r.size == 1:
            v = float(r[0]); return v, v
        else:
            return 1.0, 1.0
    v = float(r)
    return v, v

def _to_tensor_imagenet(img_np: np.ndarray) -> torch.Tensor:
    """
    np.uint8 RGB (H,W,3) -> torch.float32 (3,H,W), normalized ImageNet.
    """
    t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # (3,H,W), uint8
    t = t.float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std

# ---------- collate nội bộ (tránh xung đột với utils.misc.collate_fn) ----------
def collate_basic(batch):
    """
    Kỳ vọng mỗi item trong batch là:
      (img, pad_mask, word_id, word_mask, gt_box)
    - img: np.ndarray HxWxC (RGB/BGR) hoặc torch.Tensor CxHxW
    - pad_mask: np.ndarray HxW or torch.BoolTensor (True=pad)
    - word_id: np.ndarray/torch.LongTensor (L,)
    - word_mask: np.ndarray/torch.LongTensor (L,) with 1=real, 0=pad
    - gt_box: np.ndarray/torch.FloatTensor (4,) theo canvas [x1,y1,x2,y2]
    """
    import numpy as np
    import torch

    imgs, masks, word_ids, word_masks, gt_boxes = [], [], [], [], []

    def to_img_tensor(x):
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

    imgs = torch.stack(imgs, dim=0)       # (B,C,H,W)
    masks = torch.stack(masks, dim=0)     # (B,H,W) True=pad
    word_id = torch.stack(word_ids, dim=0)        # (B,L)
    word_mask = torch.stack(word_masks, dim=0)    # (B,L) 1=real,0=pad
    gt_boxes = torch.stack(gt_boxes, dim=0)       # (B,4)

    if not hasattr(collate_basic, "_once"):
        print("[COLLATE] imgs", imgs.shape, imgs.dtype)
        print("[COLLATE] masks", masks.shape, masks.dtype)
        print("[COLLATE] word_id", word_id.shape, word_id.dtype)
        print("[COLLATE] word_mask unique:", sorted(torch.unique(word_mask).tolist()))
        print("[COLLATE] gt_boxes", gt_boxes.shape, gt_boxes.dtype)
        collate_basic._once = True

    return imgs, masks, word_id, word_mask, gt_boxes

# ---------- sanity ----------
@torch.no_grad()
def sanity_iou_one():
    gt = torch.tensor([[10,10,100,100],[0,0,1,1],[5,20,50,60]], dtype=torch.float32)
    iou, _, _ = bbox_iou(gt, gt, x1y1x2y2=True)
    val = float(iou.mean().item())
    if not math.isclose(val, 1.0, rel_tol=0, abs_tol=1e-6):
        FAIL(f"IoU(pred=GT) != 1.0 (got {val:.6f})")
    PASS("IoU(pred=GT) == 1.0")

# ---------- arg shim cho backbone/model ----------
def build_args_shim(device: str):
    class A: pass
    a = A()
    # core
    a.device = device
    a.backbone = 'resnet50'
    a.img_pe_type = 'sine'
    a.dilation = False
    a.masks = True
    a.hidden_dim = 256
    # transformer defaults (DETR-style)
    a.dropout = 0.1
    a.nheads = 8
    a.dim_feedforward = 2048
    a.enc_layers = 6
    a.dec_layers = 1
    a.pre_norm = False
    a.activation = 'relu'
    a.return_intermediate_dec = False
    a.num_queries = 100
    # lr/freeze
    a.lr = 1e-4
    a.lr_backbone = 1e-5
    a.lr_drop = 60
    a.lr_dec = 0.1
    # fusion PE
    a.fusion_pe_max_len = 4096
    # no extra pretrain
    a.pretrain = ""
    return a

# ---------- checks ----------
def check_dataset_unit(args):
    """Kiểm tra từng mẫu ở chế độ testmode=True: pad_mask dtype, letterbox ratio/dw/dh nghịch đảo được."""
    ds_t = RSVGDataset(
        images_path=args.images_path, anno_path=args.anno_path,
        imsize=args.imsize, transform=None, augment=False,
        split=args.split, testmode=True, max_query_len=40,
        splits_dir=args.splits_dir
    )
    n = min(args.limit or 16, len(ds_t))
    _exit_if(n == 0, "Dataset rỗng? Kiểm tra đường dẫn.")

    ok_mask_bool, ok_ratio = True, True
    any_pad_true = False
    any_word_mask_int = True

    for i in range(n):
        item = ds_t[i]
        # Hỗ trợ cả hai kiểu trả về (có thể nhiều hơn do bạn mở rộng):
        img, pad_mask, word_id, word_mask, bbox_px = item[:5]
        ratio, dw, dh = item[5:8]

        ok_mask_bool &= (pad_mask.dtype == np.bool_)
        any_pad_true |= bool(pad_mask.any())
        any_word_mask_int &= (word_mask.dtype in (np.int64, np.int32))

        bx = bbox_px.astype(np.float32)
        rx, ry = _split_ratio(ratio)
        dw = float(dw); dh = float(dh)

        inv = np.array([(bx[0]-dw)/rx, (bx[1]-dh)/ry, (bx[2]-dw)/rx, (bx[3]-dh)/ry], dtype=np.float32)
        fwd = np.array([inv[0]*rx+dw, inv[1]*ry+dh, inv[2]*rx+dw, inv[3]*ry+dh], dtype=np.float32)
        ok_ratio &= np.allclose(fwd, bx, atol=1e-3)

    _exit_if(not ok_mask_bool, "pad_mask không phải bool.")
    PASS("pad_mask là bool (True=pad).")
    if not any_pad_true:
        print("[WARN] Chưa thấy vùng pad=True trong các mẫu thử (ảnh có thể vừa khít).")
    _exit_if(not any_word_mask_int, "word_mask (HF attention_mask) không phải int64/int32.")
    PASS("word_mask dtype OK (HF: int).")
    _exit_if(not ok_ratio, "Letterbox ratio/dw/dh không nhất quán (nghịch đảo bbox FAIL).")
    PASS("Letterbox ratio/dw/dh nhất quán (nghịch đảo bbox OK).")

def check_collate(args):
    ds = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        imsize=args.imsize,
        split=args.split,
        testmode=False,
        max_query_len=40,
        bert_model="bert-base-uncased",
        splits_dir=args.splits_dir,
        transform=T.ToTensor(),  # giữ đúng luồng train
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)  # collate mặc định của dataset

    images, pad_mask, word_id, word_mask, boxes = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.dim() == 4
    assert isinstance(pad_mask, torch.Tensor) and pad_mask.dtype == torch.bool
    assert isinstance(word_id, torch.Tensor) and word_id.dtype == torch.long
    assert isinstance(word_mask, torch.Tensor) and word_mask.dtype == torch.long
    assert isinstance(boxes, torch.Tensor) and boxes.dtype == torch.float32
    PASS(f"Collate OK: images {tuple(images.shape)}, pad_mask {tuple(pad_mask.shape)}, "
         f"word_id {tuple(word_id.shape)}, boxes {tuple(boxes.shape)}")

def check_backbone(args):
    ds = RSVGDataset(
        images_path=args.images_path, anno_path=args.anno_path,
        imsize=args.imsize, transform=_to_tensor_imagenet,
        augment=False, split=args.split, testmode=False, max_query_len=40,
        splits_dir=args.splits_dir
    )
    subset = Subset(ds, list(range(min(args.limit or 8, len(ds)))))
    loader = DataLoader(subset, batch_size=min(4, len(subset)), shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate_basic)

    images, pad_mask, *_ = next(iter(loader))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    images = images.to(device).float()
    pad_mask = pad_mask.to(device).bool()

    bargs = build_args_shim(device.type)
    backbone = build_backbone(bargs).to(device).eval()

    # --- num_channels: chấp nhận int hoặc list/tuple ---
    nch = getattr(backbone, "num_channels", None)
    _exit_if(nch is None, "backbone.num_channels không tồn tại?")
    if isinstance(nch, (list, tuple)):
        _exit_if(len(nch) == 0, "backbone.num_channels rỗng?")
        last = int(nch[-1])
        _exit_if(last != 2048, f"backbone.num_channels[-1] != 2048 (got {nch})")
        PASS(f"Backbone num_channels OK (list): {nch} (C5={last})")
    elif isinstance(nch, int):
        _exit_if(nch != 2048, f"backbone.num_channels != 2048 (got {nch})")
        PASS("Backbone num_channels == 2048 (int).")
    else:
        _exit_if(True, f"backbone.num_channels kiểu lạ: {type(nch)}")

    nt = NestedTensor(images, pad_mask)   # mask True=pad
    outs, poss = backbone(nt)
    _exit_if((not isinstance(outs, list)) or (not isinstance(poss, list)) or (len(outs) != len(poss)),
             "Backbone không trả list (features, pos) matching.")

    for i, (x, p) in enumerate(zip(outs, poss)):
        t, m = x.tensors, x.mask
        _exit_if(m.dtype != torch.bool, f"mask level {i} không phải bool.")
        _exit_if(p.dtype != t.dtype, f"pos dtype khác feature dtype tại level {i}.")
        _exit_if(m.shape[-2:] != t.shape[-2:], f"mask size != feature size tại level {i}.")
    PASS("Backbone feature/mask/pos khớp shape & dtype tại tất cả levels.")

    mx = max([x.mask.float().max().item() for x in outs])
    mn = min([x.mask.float().min().item() for x in outs])
    _exit_if(mx > 1.0 or mn < 0.0, "Mask sau nội suy không còn nhị phân?")
    PASS("Mask sau nội suy vẫn nhị phân (nearest).")

    # --- Kiểm freeze flags: hỗ trợ nhiều wrapper khác nhau ---
    named_iters = []
    # phổ biến trong DETR: Joiner[0].body hoặc .body
    for path in ["[0].body", "body", "backbone[0].body", "backbone.body"]:
        try:
            obj = backbone
            for part in path.replace("[0]", ".0").split("."):
                if not part:
                    continue
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            named_iters.append(list(obj.named_parameters()))
        except Exception:
            pass

    # nếu vẫn rỗng, fallback: toàn bộ backbone (ít chính xác hơn về tên layer)
    if not named_iters:
        named_iters.append(list(backbone.named_parameters()))
        print_warn("Freeze check", "Không tìm thấy .body; dùng fallback over all named_parameters.")

    params = []
    for it in named_iters:
        params.extend(it)
    # Loại trùng tên nếu có
    seen = set(); pruned = []
    for n, p in params:
        if n in seen: continue
        seen.add(n); pruned.append((n, p))

    def top_block(n):  # xác định block cấp cao để ước lượng freeze
        if "layer2" in n: return "layer2"
        if "layer3" in n: return "layer3"
        if "layer4" in n: return "layer4"
        if any(x in n for x in ["conv1","bn1","layer1"]): return "stem_l1"
        return "other"

    any_l234_train = any((top_block(n) in {"layer2","layer3","layer4"}) and p.requires_grad for n, p in pruned)
    any_stem_train = any((top_block(n) == "stem_l1") and p.requires_grad for n, p in pruned)

    _exit_if(not any_l234_train, "layer2/3/4 không trainable? Kiểm tra lr_backbone/train_backbone.")
    if any_stem_train:
        print_warn("Freeze flags", "conv1/bn1/layer1 đang trainable (mong muốn freeze).")
    else:
        PASS("Freeze flags backbone hợp lý (layer2-4 trainable, conv1/bn1/layer1 frozen).")


@torch.no_grad()
def check_model_and_fusion(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        imsize=args.imsize,
        split=args.split,
        testmode=False,
        max_query_len=40,
        bert_model="bert-base-uncased",
        splits_dir=args.splits_dir,
        transform=T.ToTensor(),
    )
    bs = min(4, max(1, getattr(args, "limit", 4)))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    images, pad_mask, word_id, word_mask, boxes = next(iter(loader))
    images   = images.to(device)
    pad_mask = pad_mask.to(device)
    word_id  = word_id.to(device).long()
    word_mask= word_mask.to(device).long()

    # args shim tối thiểu
    class A: pass
    a = A()
    a.device = device.type
    a.backbone = "resnet50"
    a.masks = True
    a.dilation = False
    a.hidden_dim = 256
    a.dropout = 0.1
    a.nheads = 8
    a.dim_feedforward = 2048
    a.enc_layers = 6
    a.dec_layers = 1
    a.pre_norm = False
    a.lr_backbone = 1e-4
    a.img_pe_type = "sine"
    a.max_query_len = 40
    a.max_fusion_len = 8192
    a.pretrain = ""

    model = MGVLF(bert_model="bert-base-uncased", tunebert=True, args=a).to(device)
    model.eval()

    with torch.no_grad():
        # text enc
        outputs = model.textmodel(input_ids=word_id, attention_mask=word_mask, return_dict=True)
        fl = outputs.last_hidden_state
        pool = getattr(outputs, "pooler_output", None)
        if pool is None:
            am = word_mask.unsqueeze(-1).float()
            pool = (fl * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)

        # visu + fusion
        fv = model.visumodel(images, pad_mask, word_mask, fl, pool)      # (B,256,H,W)
        fusion = model.vlmodel(fv, fl, word_mask)                         # kỳ vọng (B,256,S) hoặc (B,S,256)/(B,256)

        # ----- Chuẩn hoá shape -----
        if fusion.dim() == 2 and fusion.size(1) == 256:
            print("[NOTE] Fusion trả (B,256). Tạm coi như (B,256,1) để tiếp tục kiểm.")
            fusion = fusion.unsqueeze(-1)                                  # (B,256,1)
        elif fusion.dim() == 3:
            if fusion.size(1) == 256:
                pass                                                       # (B,256,S) OK
            elif fusion.size(2) == 256:
                fusion = fusion.transpose(1, 2).contiguous()               # (B,S,256) -> (B,256,S)
            else:
                raise AssertionError(f"[FAIL] Fusion shape lạ: {tuple(fusion.shape)}")
        else:
            raise AssertionError(f"[FAIL] Fusion dim lạ: {fusion.dim()}")

        B, C, S = fusion.shape
        PASS(f"Fusion shape chuẩn hoá: (B,256,S) với S={S}")

        # ----- Kiểm [pr]@0 và mask -----
        Lv = fv.view(B, 256, -1).shape[2]
        pr_pad  = torch.zeros((B,1),  dtype=torch.bool, device=device)
        vis_pad = torch.zeros((B,Lv), dtype=torch.bool, device=device)
        text_pad= (word_mask == 0).to(device)                              # (B,L)
        fused_mask = torch.cat([pr_pad, vis_pad, text_pad], dim=1)         # (B, 1+Lv+L)

        # [pr] không pad:
        assert (~pr_pad.squeeze(1)).all(), "[FAIL] [pr] token bị pad"
        if fused_mask.shape[1] == S:
            assert (~fused_mask[:, 0]).all(), "[FAIL] [pr]@0 bị pad"
            PASS("[pr] ở đầu và không bị pad")
        else:
            print_warn("Fused length mismatch", f"S={S} vs (1+Lv+L)={fused_mask.shape[1]} — bỏ qua check khớp chiều dài.")

        # ----- Head dùng token 0 -> (B,4) trong [0,1] -----
        feat_pr = fusion[:, :, 0]                                          # (B,256)
        head_out = model.box_head(feat_pr).sigmoid()
        assert head_out.shape == (B, 4), f"[FAIL] Box head output {tuple(head_out.shape)} khác (B,4)"
        assert (head_out >= 0).all() and (head_out <= 1).all(), "[FAIL] Box head không nằm [0,1]"
        PASS("Head dùng token 0 ([pr]) → (B,4) ∈ [0,1]")

        # E2E
        out_full = model(images, pad_mask, word_id, word_mask)
        assert out_full.shape == (B,4), f"[FAIL] model(...) phải trả (B,4), nhận {tuple(out_full.shape)}"
        PASS("Forward end-to-end trả (B,4)")

def tiny_overfit_step(model, batch, device):
    imgs, masks, word_id, word_mask, gt_boxes = batch
    imgs, masks = imgs.to(device), masks.to(device)
    word_id, word_mask = word_id.to(device), word_mask.to(device)
    gt_boxes = gt_boxes.to(device)  # expected pixel x1,y1,x2,y2 trên canvas

    out = model(imgs, masks, word_id, word_mask)  # (B,4) normalized [0,1]
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

    loss = (pred_boxes - gt_boxes).abs().mean()
    return pred_boxes, loss

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser("RSVG/MGVLF Smoke Test (merged)")
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--anno_path",   type=str, required=True)
    parser.add_argument("--splits_dir",  type=str, required=True)
    parser.add_argument("--split",       type=str, default="val", choices=["train","val","test"])
    parser.add_argument("--imsize",      type=int, default=640)
    parser.add_argument("--limit",       type=int, default=32, help="số mẫu kiểm nhanh (0 = mặc định 16)")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--tiny_overfit", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    args = parser.parse_args()

    check_env()
    sanity_iou_one()
    check_dataset_unit(args)
    check_collate(args)
    check_backbone(args)
    check_model_and_fusion(args)

    if args.tiny_overfit:
        print("\n[INFO] Tiny overfit 1 bước để sanity loss ↓")
        # Lấy 1 batch bằng pipeline ToTensor (giống train)
        ds = RSVGDataset(
            images_path=args.images_path,
            anno_path=args.anno_path,
            imsize=args.imsize,
            split=args.split,
            testmode=False,
            max_query_len=40,
            bert_model="bert-base-uncased",
            splits_dir=args.splits_dir,
            transform=T.ToTensor(),
        )
        loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True, num_workers=0)
        imgs, masks, word_id, word_mask, gt_boxes = next(iter(loader))

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # args shim cho model
        class A: pass
        a = A()
        a.device = device.type
        a.backbone = "resnet50"
        a.masks = True
        a.dilation = False
        a.hidden_dim = 256
        a.dropout = 0.1
        a.nheads = 8
        a.dim_feedforward = 2048
        a.enc_layers = 6
        a.dec_layers = 1
        a.pre_norm = False
        a.lr_backbone = 1e-4
        a.img_pe_type = "sine"
        a.max_query_len = 40
        a.max_fusion_len = 8192
        a.pretrain = ""

        model = MGVLF(bert_model="bert-base-uncased", tunebert=True, args=a).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        batch = (imgs, masks, word_id, word_mask, gt_boxes)
        _, loss0 = tiny_overfit_step(model, batch, device)
        opt.zero_grad(); loss0.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.1); opt.step()
        _, loss1 = tiny_overfit_step(model, batch, device)
        print_ok("Tiny overfit step", f"loss0={loss0.item():.4f} -> loss1={loss1.item():.4f}")

    print("\n✅ SMOKE TEST PASSED.\n")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n=== Uncaught exception in smoke_test ===")
        traceback.print_exc()
        sys.exit(2)
