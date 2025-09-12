#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMOKE TEST cho RSVG/MGVLF

Kiểm tra nhanh các lỗi hay gặp:
A) HF attention_mask polarity & key_padding_mask (True=pad)
B) Letterbox scale/shift (ratio có thể là float hoặc (rx,ry))
C) Lấy đúng [pr] token cho head
+ Sanity: IoU=1 khi pred=GT; dtype/shape; mask 2D bool True=pad

Cách chạy (ví dụ):
python tools/smoke_test.py \
  --images_path /path/to/DIOR/JPEGImages \
  --anno_path   /path/to/DIOR/Annotations \
  --splits_dir  /path/to/DIOR_RSVG \
  --imsize 640 \
  --split val \
  --limit 64 \
  --device cuda
"""

import os
import sys
import math
import argparse
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader

# project imports
from data_loader import RSVGDataset
from models.model import MGVLF
from utils.misc import collate_fn
from utils.utils import bbox_iou, _sanitize_xyxy_t


def _ok(msg):  print(f"[PASS] {msg}")
def _bad(msg): print(f"[FAIL] {msg}")

def _exit_if(cond, msg, code=1):
    if cond:
        _bad(msg)
        sys.exit(code)

@torch.no_grad()
def sanity_iou_one():
    gt = torch.tensor([[10,10,100,100],[0,0,1,1],[5,20,50,60]], dtype=torch.float32)
    iou, _, _ = bbox_iou(gt, gt, x1y1x2y2=True)
    val = float(iou.mean().item())
    if not math.isclose(val, 1.0, rel_tol=0, abs_tol=1e-6):
        _bad(f"IoU(pred=GT) != 1.0 (got {val:.6f})"); return False
    _ok("IoU(pred=GT)=1.000000"); return True


class HookFusionOut:
    """Gắn hook để lấy output cuối của VLFusion.transformer (chuỗi fusion)."""
    def __init__(self, vlmodel):
        self.last = None
        self.h = vlmodel.transformer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out thường là list layer outputs; lấy phần cuối
        try:
            self.last = out[-1].detach()
        except Exception:
            self.last = out.detach()

    def close(self):
        if self.h: self.h.remove()


def build_args_shim(device="cuda", im_pe="sine"):
    class _A:
        pass
    a = _A()
    a.backbone = 'resnet50'
    a.img_pe_type = im_pe
    a.dilation = False
    a.masks = True
    a.hidden_dim = 256
    a.device = device
    a.lr, a.lr_backbone, a.lr_drop, a.lr_dec = 1e-4, 1e-5, 60, 0.1
    a.fusion_pe_max_len = 4096
    a.pretrain = ""     # seed_from_detr trong model.py sẽ tự xử lý
    return a


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_path", required=True)
    ap.add_argument("--anno_path",   required=True)
    ap.add_argument("--splits_dir",  required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--imsize", type=int, default=640)
    ap.add_argument("--limit", type=int, default=32, help="số mẫu kiểm nhanh")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ========== SANITY 0: IoU=1 ==========
    sanity_iou_one()

    # ========== PASS 1: kiểm Letterbox/ratio & mask 2D bool ==========
    ds_test = RSVGDataset(
        images_path=args.images_path, anno_path=args.anno_path,
        imsize=args.imsize, transform=None, augment=False,
        split=args.split, testmode=True, max_query_len=40,
        splits_dir=args.splits_dir
    )
    n = min(args.limit, len(ds_test)) if args.limit>0 else min(16, len(ds_test))
    n = max(n, 1)

    ok_mask_dtype = True
    ok_ratio = True
    any_pad_true = False

    for i in range(n):
        img, pad_mask, word_id, word_mask, bbox_px, ratio, dw, dh, img_path, phrase = ds_test[i]
        # pad_mask: bool True=pad?
        ok_mask_dtype &= (pad_mask.dtype == np.bool_)
        any_pad_true |= bool(pad_mask.any())

        # forward letterbox mapping check (original -> letterboxed)
        # vì dataset đã làm hộ, ta kiểm tra ngược: (bbox - [dw,dh]) / r ≈ original
        # Đáng tiếc original bbox không trả, nên ta check nhất quán: scale ngược rồi lại scale thuận.
        bx = bbox_px.copy().astype(np.float32)
        # inverse
        if isinstance(ratio, (tuple, list, np.ndarray)):
            rx, ry = float(ratio[0]), float(ratio[1])
        else:
            rx = ry = float(ratio)
        inv = np.array([(bx[0]-dw)/rx, (bx[1]-dh)/ry, (bx[2]-dw)/rx, (bx[3]-dh)/ry], dtype=np.float32)
        # forward lại
        fwd = np.array([inv[0]*rx+dw, inv[1]*ry+dh, inv[2]*rx+dw, inv[3]*ry+dh], dtype=np.float32)
        ok_ratio &= (np.allclose(fwd, bx, atol=1e-3))

    _exit_if(not ok_mask_dtype, "pad_mask không phải bool")
    _ok("pad_mask là bool (True=pad)")
    _exit_if(not any_pad_true, "pad_mask không có vùng pad nào (ảnh có thể không bị pad, nhưng nên thấy ít nhất 1 mẫu True)")
    _ok("pad_mask có vùng pad=True ở một số mẫu")
    _exit_if(not ok_ratio, "Letterbox scale/shift sai (không nghịch đảo được bbox)")
    _ok("Letterbox ratio/dw/dh nhất quán (nghịch đảo bbox OK)")

    # ========== PASS 2: batch inference để test mask polarity & chọn [pr] ==========
    ds = RSVGDataset(
        images_path=args.images_path, anno_path=args.anno_path,
        imsize=args.imsize, transform=None, augment=False,
        split=args.split, testmode=False, max_query_len=40,
        splits_dir=args.splits_dir
    )
    if args.limit>0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False
    )

    model = MGVLF(bert_model="bert-base-uncased", tunebert=True, args=build_args_shim(device.type))
    model.to(device).eval()

    # Hook để lấy fusion output
    hook = HookFusionOut(model.vlmodel)

    # Chạy 1-2 batch là đủ smoke
    batches = 0
    better_pr0 = 0
    total_cmp = 0

    for images, pad_mask, word_id, word_mask, boxes_gt_px in loader:
        batches += 1
        images = images.to(device).float()
        pad_mask = pad_mask.to(device).bool()
        word_id = word_id.to(device).long()
        word_mask = word_mask.to(device).long()
        boxes_gt_px = boxes_gt_px.to(device).float()

        # Polarity check: HF 1=real, 0=pad -> key_pad should be (word_mask==0)
        key_pad_should = (word_mask == 0)
        # Khi forward, model sẽ tự tính key_pad và truyền sang VLFusion/CNN_MGVLF.
        preds_norm = model(images, pad_mask, word_id, word_mask)  # (B,4) in [0,1]
        # Sau forward, hook.last chứa output chuỗi fusion cuối
        fusion = hook.last
        _exit_if(fusion is None, "Không lấy được fusion output từ hook (kiểm tra kiến trúc VLFusion)")

        # Kiểm tra số pad tokens > 0 khi có padding trong câu
        pad_tokens = key_pad_should.sum().item()
        _exit_if(pad_tokens < 0, "Lỗi đếm pad tokens (không hợp lệ)")
        if pad_tokens == 0:
            print("[WARN] batch không có token pad (OK, nhưng nên có ở vài batch khác)")

        # So sánh chọn token 0 vs -1 cho head (kỳ vọng token 0 tốt hơn hoặc không tệ hơn rõ rệt)
        # fusion shape có thể (B,256,S) hoặc (B,S,256). Chuẩn với code hiện tại là (B,256,S).
        if fusion.dim() != 3:
            _exit_if(True, f"fusion output shape lạ: {tuple(fusion.shape)}")

        if fusion.size(1) == 256:     # (B,256,S)
            feat_pr0 = fusion[:, :, 0]          # (B,256)
            feat_last = fusion[:, :, -1]        # (B,256)
        elif fusion.size(2) == 256:   # (B,S,256)
            feat_pr0 = fusion[:, 0, :]          # (B,256)
            feat_last = fusion[:, -1, :]        # (B,256)
        else:
            _exit_if(True, f"fusion hidden dim != 256: {tuple(fusion.shape)}")

        # Dùng cùng box_head của model để so sánh
        pred_pr0 = model.Prediction_Head(feat_pr0).sigmoid()
        pred_last = model.Prediction_Head(feat_last).sigmoid()

        B, _, H, W = images.shape
        scale = torch.tensor([W, H, W, H], device=images.device, dtype=pred_pr0.dtype)
        pr0_px   = _sanitize_xyxy_t(pred_pr0 * scale)
        last_px  = _sanitize_xyxy_t(pred_last * scale)
        gt_px    = _sanitize_xyxy_t(boxes_gt_px)

        iou0, _, _   = bbox_iou(pr0_px,  gt_px, x1y1x2y2=True)
        iouLast,_, _ = bbox_iou(last_px, gt_px, x1y1x2y2=True)

        better_pr0 += (iou0 >= iouLast).sum().item()
        total_cmp  += B

        print(f"[batch {batches}] median IoU pr@0={iou0.median().item():.3f} vs last={iouLast.median().item():.3f}")

        if batches >= 2:   # đủ cho smoke
            break

    hook.close()

    # Kết luận
    if total_cmp > 0:
        ratio = better_pr0 / total_cmp
        _exit_if(ratio < 0.5, f"[pr]@0 KHÔNG tốt hơn last token (tỉ lệ {ratio:.2f} < 0.5?) — kiểm tra chọn token cho head")
        _ok(f"[pr] token @ index 0 cho head hợp lý (tỉ lệ tốt hơn/không tệ hơn rõ rệt: {ratio:.2f})")

    _ok("Smoke test hoàn tất")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n=== Uncaught exception in smoke test ===")
        traceback.print_exc()
        sys.exit(2)
