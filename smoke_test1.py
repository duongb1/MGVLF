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

def collate_fn(batch):
    # batch: list of tuples theo output của RSVGDataset
    imgs, masks, word_id, word_mask, gt_boxes = [], [], [], [], []
    for b in batch:
        # chỉnh nếu RSVGDataset trả khác thứ tự:
        # ví dụ bạn đang trả (img, pad_mask, word_id, word_mask, box)
        imgs.append(b[0])
        masks.append(b[1])
        word_id.append(b[2])
        word_mask.append(b[3])
        gt_boxes.append(b[4])
    imgs = torch.stack(imgs, 0).float()
    masks = torch.stack(masks, 0).bool()
    word_id = torch.stack(word_id, 0).long()
    word_mask = torch.stack(word_mask, 0).long()
    gt_boxes = torch.stack(gt_boxes, 0).float()
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
    # gọi nhánh backbone bên trong visumodel nếu có expose; fallback: chạy visumodel ở chế độ dry
    try:
        # Nếu visumodel có thuộc tính 'backbone' kiểu Joiner
        back = model_visu.backbone
        from utils.misc import NestedTensor
        samples = NestedTensor(imgs, masks)  # (B,3,H,W) + (B,H,W)
        feats, out_masks, pos = back(samples)
        print_ok("Backbone forward")
        n = len(feats)
        ch = [f.shape[1] for f in feats]
        print(f"[DBG] backbone maps={n}, channels={ch}")
        expect = 3 if (args.aux_loss or args.masks) else 1
        if n != expect:
            print_warn("Unexpected #feature maps", f"got {n}, expect {expect}")
    except Exception as e:
        print_warn("Backbone debug skipped", str(e))

def tiny_overfit_step(model, batch, device):
    imgs, masks, word_id, word_mask, gt_boxes = batch
    imgs = imgs.to(device)
    masks = masks.to(device)
    word_id = word_id.to(device)
    word_mask = word_mask.to(device)
    gt_boxes = gt_boxes.to(device)

    out = model(imgs, masks, word_id, word_mask)  # (B,4) sigmoid
    # simple L1 to GT center/w/h (giả sử GT đã cùng canvas)
    loss = torch.abs(out - gt_boxes).mean()
    return out, loss

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
    model.eval()
    print_ok("Model built")

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
