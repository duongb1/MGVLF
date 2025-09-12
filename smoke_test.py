#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMOKE TEST cho RSVG/MGVLF: Data pipeline + Backbone + Fusion [pr] token.

Chạy (ví dụ Kaggle):
python smoke_test.py \
  --images_path "/kaggle/input/dior-rsvg/DIOR_RSVG/JPEGImages" \
  --anno_path   "/kaggle/input/dior-rsvg/DIOR_RSVG/Annotations" \
  --splits_dir  "/kaggle/input/dior-rsvg/DIOR_RSVG" \
  --split val \
  --imsize 640 \
  --limit 32 \
  --device cuda
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tránh cảnh báo fork của HF tokenizers

from torchvision import transforms as T
_SMOKE_TF = T.ToTensor()  # chỉ dùng cho smoke test

import sys
import math
import argparse
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# local repo imports
from data_loader import RSVGDataset
from utils.misc import collate_fn, NestedTensor
from utils.utils import bbox_iou
from models.backbone import build_backbone
from models.model import MGVLF


# ---------- helpers ----------
def PASS(msg): print(f"[PASS] {msg}")
def FAIL(msg): print(f"[FAIL] {msg}") or sys.exit(1)
def _exit_if(cond, msg, code=1):
    if cond:
        FAIL(msg)
        sys.exit(code)

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
    # ---- core ----
    a.device = device
    a.backbone = 'resnet50'
    a.img_pe_type = 'sine'
    a.dilation = False
    a.masks = True
    a.hidden_dim = 256

    # ---- transformer defaults (DETR-style) ----
    a.dropout = 0.1
    a.nheads = 8
    a.dim_feedforward = 2048
    a.enc_layers = 6          # encoder layers cho EN & VLFusion
    a.dec_layers = 1          # decoder layers cho DE (refine 1 lượt)
    a.pre_norm = False
    a.activation = 'relu'
    a.return_intermediate_dec = False
    a.num_queries = 100       # nếu builder có dùng, không hại

    # ---- lr/freeze ----
    a.lr = 1e-4
    a.lr_backbone = 1e-5
    a.lr_drop = 60
    a.lr_dec = 0.1

    # ---- fusion PE ----
    a.fusion_pe_max_len = 4096

    # ---- no extra pretrain here ----
    a.pretrain = ""
    return a



# ---------- hook lấy output chuỗi cuối của VLFusion ----------
class HookFusionOut:
    def __init__(self, vlmodel):
        self.last = None
        self.h = vlmodel.transformer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        try:
            self.last = out[-1].detach()
        except Exception:
            self.last = out.detach()

    def close(self):
        if self.h: self.h.remove()


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
        img, pad_mask, word_id, word_mask, bbox_px, ratio, dw, dh, *_ = ds_t[i]
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
    import torch
    from torch.utils.data import DataLoader
    from data_loader import RSVGDataset

    ds = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        imsize=args.imsize,
        split=args.split,
        testmode=False,
        max_query_len=40,
        bert_model="bert-base-uncased",
        splits_dir=args.splits_dir,
        transform=_SMOKE_TF,  # <— thêm
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)  # <— bỏ collate_fn

    images, pad_mask, word_id, word_mask, boxes = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.dim() == 4
    assert isinstance(pad_mask, torch.Tensor) and pad_mask.dtype == torch.bool
    assert isinstance(word_id, torch.Tensor) and word_id.dtype == torch.long
    assert isinstance(word_mask, torch.Tensor) and word_mask.dtype == torch.long
    assert isinstance(boxes, torch.Tensor) and boxes.dtype == torch.float32
    print(f"[PASS] Collate OK: images {tuple(images.shape)}, pad_mask {tuple(pad_mask.shape)}, "
          f"word_id {tuple(word_id.shape)}, boxes {tuple(boxes.shape)}")


def check_backbone(args):
    ds = RSVGDataset(
        images_path=args.images_path, anno_path=args.anno_path,
        imsize=args.imsize, transform=_to_tensor_imagenet,  # <--- CHUYỂN ẢNH SANG TENSOR
        augment=False, split=args.split, testmode=False, max_query_len=40,
        splits_dir=args.splits_dir
    )
    subset = Subset(ds, list(range(min(args.limit or 8, len(ds)))))
    loader = DataLoader(subset, batch_size=min(4, len(subset)), shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate_fn)

    images, pad_mask, *_ = next(iter(loader))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    images = images.to(device).float()
    pad_mask = pad_mask.to(device).bool()

    bargs = build_args_shim(device.type)
    backbone = build_backbone(bargs).to(device).eval()

    _exit_if(getattr(backbone, "num_channels", None) != 2048,
             f"backbone.num_channels != 2048 (got {getattr(backbone,'num_channels',None)})")
    PASS("Backbone num_channels == 2048 (ResNet50).")

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

    trainable = [(n, p.requires_grad) for n, p in backbone[0].body.named_parameters()]

    seg = lambda n: n.split('.', 1)[0]  # only the top-level block name

    any_l2l4_true = any(seg(n) in {'layer2', 'layer3', 'layer4'} and rg for n, rg in trainable)
    any_c1l1_true = any(seg(n) in {'conv1', 'bn1', 'layer1'} and rg for n, rg in trainable)
    
    # Optional debug to see what's trainable
    print("[DBG init trainable]", [n for n, rg in trainable if rg and seg(n) in {'conv1','bn1','layer1'}])
    print("[DBG l234 trainable]", [n for n, rg in trainable if rg and seg(n) in {'layer2','layer3','layer4'}][:10])
    
    _exit_if(not any_l2l4_true, "layer2/3/4 không trainable? Kiểm tra lr_backbone/train_backbone.")
    _exit_if(any_c1l1_true, "conv1/bn1/layer1 đang trainable (mong muốn freeze).")
    PASS("Freeze flags backbone hợp lý (layer2-4 trainable, conv1/bn1/layer1 frozen).")


@torch.no_grad()
def check_model_and_fusion(args):
    import torch
    from torch.utils.data import DataLoader
    from data_loader import RSVGDataset
    from models.model import MGVLF

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
        transform=_SMOKE_TF,  # <— thêm
    )
    bs = min(4, max(1, getattr(args, "limit", 4)))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)  # <— bỏ collate_fn

    images, pad_mask, word_id, word_mask, boxes = next(iter(loader))
    images   = images.to(device)                    # (B,3,H,W)
    pad_mask = pad_mask.to(device)                  # (B,H,W) True=pad
    word_id  = torch.as_tensor(word_id,  device=device).long()   # (B,L)
    word_mask= torch.as_tensor(word_mask,device=device).long()   # (B,L)

    # args shim cho model/transformer
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
    a.dec_layers = 1   # DE 1 layer
    a.lr_backbone = 1e-4  # để freeze conv1/bn1/layer1, train layer2-4
    a.img_pe_type = "sine"
    a.max_query_len = 40
    a.max_fusion_len = 8192
    a.pretrain = ""    # không nạp ckpt tùy chọn

    model = MGVLF(bert_model="bert-base-uncased", tunebert=True, args=a).to(device)
    model.eval()

    with torch.no_grad():
        # chạy từng nhánh để soi fusion shape
        outputs = model.textmodel(input_ids=word_id, attention_mask=word_mask, return_dict=True)
        fl = outputs.last_hidden_state                                        # (B,L,768)
        pool = getattr(outputs, "pooler_output", None)
        if pool is None:
            am = word_mask.unsqueeze(-1).float()
            pool = (fl * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)

        fv = model.visumodel(images, pad_mask, word_mask, fl, pool)          # (B,256,H,W)
        fusion = model.vlmodel(fv, fl, word_mask)                             # EXPECT: (B,256,S)

        # ---- Checks ----
        assert fusion.dim() == 3 and fusion.size(1) == 256, \
            f"[FAIL] Fusion phải là (B,256,S), nhận {tuple(fusion.shape)}"
        B, C, S = fusion.shape
        print(f"[PASS] Fusion shape (B,256,S) với S={S}")

        # [pr] ở đầu chuỗi và không bị pad (mask vị trí 0 phải False)
        # tạo mask như trong VLFusion: [pr | vis | text]
        Lv = fv.view(B, 256, -1).shape[2]
        pr_pad  = torch.zeros((B,1),  dtype=torch.bool, device=device)
        vis_pad = torch.zeros((B,Lv), dtype=torch.bool, device=device)
        text_pad= (word_mask == 0).to(device)                                 # (B,L) True=pad
        fused_mask = torch.cat([pr_pad, vis_pad, text_pad], dim=1)            # (B,S_expected)
        assert fused_mask.shape[1] == S, f"[FAIL] mask hợp thành có S={fused_mask.shape[1]} khác fusion.S={S}"
        assert (~fused_mask[:,0]).all(), "[FAIL] [pr] token bị pad"
        print("[PASS] [pr] token ở đầu và không bị pad")

        # Head lấy token 0 -> (B,4) trong [0,1]
        feat_pr = fusion[:, :, 0]                                             # (B,256)
        head_out = model.box_head(feat_pr).sigmoid()
        assert head_out.shape == (B,4), f"[FAIL] Box head output phải (B,4), nhận {tuple(head_out.shape)}"
        assert (head_out >= 0).all() and (head_out <= 1).all(), "[FAIL] Box head không nằm trong [0,1]"
        print("[PASS] Head dùng token 0 ([pr]) → (B,4) trong [0,1]")

        # Thử forward end-to-end cũng trả (B,4)
        out_full = model(images, pad_mask, word_id, word_mask)
        assert out_full.shape == (B,4), f"[FAIL] model(...) phải trả (B,4), nhận {tuple(out_full.shape)}"
        print("[PASS] Forward end-to-end trả (B,4)")



# ---------- main ----------
def main():
    parser = argparse.ArgumentParser("RSVG/MGVLF Smoke Test")
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--anno_path",   type=str, required=True)
    parser.add_argument("--splits_dir",  type=str, required=True)
    parser.add_argument("--split",       type=str, default="val", choices=["train","val","test"])
    parser.add_argument("--imsize",      type=int, default=640)
    parser.add_argument("--limit",       type=int, default=32, help="số mẫu kiểm nhanh (0 = mặc định 16)")
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    sanity_iou_one()
    check_dataset_unit(args)   # testmode=True, không cần transform
    check_collate(args)        # dùng _to_tensor_imagenet
    check_backbone(args)       # dùng _to_tensor_imagenet
    check_model_and_fusion(args)  # dùng _to_tensor_imagenet

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
