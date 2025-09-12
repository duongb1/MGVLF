# models/model.py
from collections import OrderedDict
from typing import Optional
import os, urllib.request, types

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CNN_MGVLF import build_VLFusion, build_CNN_MGVLF
from transformers import AutoModel, AutoConfig


# ==== DETR seeding helpers (copy weights giữa các layer tương thích) ====
def _copy_enc_layer(dst, src):
    with torch.no_grad():
        # self-attn
        dst.self_attn.in_proj_weight.copy_(src.self_attn.in_proj_weight)
        dst.self_attn.in_proj_bias.copy_(src.self_attn.in_proj_bias)
        dst.self_attn.out_proj.weight.copy_(src.self_attn.out_proj.weight)
        dst.self_attn.out_proj.bias.copy_(src.self_attn.out_proj.bias)
        # FFN
        dst.linear1.weight.copy_(src.linear1.weight); dst.linear1.bias.copy_(src.linear1.bias)
        dst.linear2.weight.copy_(src.linear2.weight); dst.linear2.bias.copy_(src.linear2.bias)
        # Norms
        dst.norm1.weight.copy_(src.norm1.weight); dst.norm1.bias.copy_(src.norm1.bias)
        dst.norm2.weight.copy_(src.norm2.weight); dst.norm2.bias.copy_(src.norm2.bias)

def _copy_dec_layer(dst, src):
    with torch.no_grad():
        # self-attn
        dst.self_attn.in_proj_weight.copy_(src.self_attn.in_proj_weight)
        dst.self_attn.in_proj_bias.copy_(src.self_attn.in_proj_bias)
        dst.self_attn.out_proj.weight.copy_(src.self_attn.out_proj.weight)
        dst.self_attn.out_proj.bias.copy_(src.self_attn.out_proj.bias)
        # cross-attn
        dst.multihead_attn.in_proj_weight.copy_(src.multihead_attn.in_proj_weight)
        dst.multihead_attn.in_proj_bias.copy_(src.multihead_attn.in_proj_bias)
        dst.multihead_attn.out_proj.weight.copy_(src.multihead_attn.out_proj.weight)
        dst.multihead_attn.out_proj.bias.copy_(src.multihead_attn.out_proj.bias)
        # FFN
        dst.linear1.weight.copy_(src.linear1.weight); dst.linear1.bias.copy_(src.linear1.bias)
        dst.linear2.weight.copy_(src.linear2.weight); dst.linear2.bias.copy_(src.linear2.bias)
        # Norms
        dst.norm1.weight.copy_(src.norm1.weight); dst.norm1.bias.copy_(src.norm1.bias)
        dst.norm2.weight.copy_(src.norm2.weight); dst.norm2.bias.copy_(src.norm2.bias)
        dst.norm3.weight.copy_(src.norm3.weight); dst.norm3.bias.copy_(src.norm3.bias)


def load_weights(model: nn.Module, load_path: str) -> nn.Module:
    """
    Nạp trọng số linh hoạt từ nhiều định dạng checkpoint:
    - {'model': state_dict} hoặc {'state_dict': state_dict} hoặc state_dict thuần.
    Chỉ copy những key trùng tên và cùng shape.
    """
    ckpt = torch.load(load_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    current = model.state_dict()
    for k in current.keys():
        if k in state and current[k].shape == state[k].shape:
            current[k] = state[k]
    model.load_state_dict(current, strict=False)
    del current, state, ckpt
    torch.cuda.empty_cache()
    return model


class MGVLF(nn.Module):
    """
    Text encoder (BERT: word-level + 1 sentence) →
    CNN_MGVLF (DE refine fv1 bằng multi-scale vis + text) →
    VLFusion (encoder với word tokens + [pr]) →
    Box head (cx,cy,w,h) ∈ [0,1]
    """
    def __init__(self, bert_model: str = 'bert-base-uncased', tunebert: bool = True, args: Optional[object] = None):
        super(MGVLF, self).__init__()
        self.tunebert = tunebert
        self.args = args

        # -------- Text model (HF Transformers) --------
        config = AutoConfig.from_pretrained(
            bert_model,
            output_hidden_states=True,
            add_pooling_layer=True,
        )
        self.textmodel = AutoModel.from_pretrained(bert_model, config=config)

        if not self.tunebert:
            for p in self.textmodel.parameters():
                p.requires_grad = False

        # -------- Visual model (CNN branch) --------
        self.visumodel = build_CNN_MGVLF(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.visumodel = load_weights(self.visumodel, args.pretrain)

        # -------- Fusion model --------
        self.vlmodel = build_VLFusion(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.vlmodel = load_weights(self.vlmodel, args.pretrain)

        # ===== SEED TỪ DETR (auto download) =====
        try:
            os.makedirs("pretrained", exist_ok=True)
            ckpt = "./pretrained/detr-r50-e632da11.pth"
            if not os.path.isfile(ckpt):
                url = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
                print(f"[DETR seed] downloading from {url} ...")
                urllib.request.urlretrieve(url, ckpt)
            self.seed_from_detr(ckpt)
            print("[DETR seed] done.")
        except Exception as e:
            print(f"[DETR seed] skipped: {e}")

        # -------- Localization Head --------
        self.box_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        self.Prediction_Head = self.box_head  # alias

        for m in self.box_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def seed_from_detr(self, ckpt_path: str, use_last_dec_layer: bool = False):
        """
        Copy weight từ checkpoint DETR ResNet-50 (COCO pretrained).
        """
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model", sd.get("state_dict", sd))

        # 1) input_proj
        with torch.no_grad():
            self.visumodel.input_proj.weight.copy_(sd["input_proj.weight"])
            self.visumodel.input_proj.bias.copy_(sd["input_proj.bias"])

        # 2) encoder layers
        for i, (dst_visu, dst_vlf) in enumerate(
            zip(self.visumodel.transformer.encoder.layers,
                self.vlmodel.transformer.encoder.layers)
        ):
            prefix = f"transformer.encoder.layers.{i}"
            src = types.SimpleNamespace(
                self_attn=types.SimpleNamespace(
                    in_proj_weight=sd[f"{prefix}.self_attn.in_proj_weight"],
                    in_proj_bias=sd[f"{prefix}.self_attn.in_proj_bias"],
                    out_proj=types.SimpleNamespace(
                        weight=sd[f"{prefix}.self_attn.out_proj.weight"],
                        bias=sd[f"{prefix}.self_attn.out_proj.bias"],
                    ),
                ),
                linear1=types.SimpleNamespace(
                    weight=sd[f"{prefix}.linear1.weight"],
                    bias=sd[f"{prefix}.linear1.bias"],
                ),
                linear2=types.SimpleNamespace(
                    weight=sd[f"{prefix}.linear2.weight"],
                    bias=sd[f"{prefix}.linear2.bias"],
                ),
                norm1=types.SimpleNamespace(
                    weight=sd[f"{prefix}.norm1.weight"],
                    bias=sd[f"{prefix}.norm1.bias"],
                ),
                norm2=types.SimpleNamespace(
                    weight=sd[f"{prefix}.norm2.weight"],
                    bias=sd[f"{prefix}.norm2.bias"],
                ),
            )
            _copy_enc_layer(dst_visu, src)
            _copy_enc_layer(dst_vlf, src)

        # 3) decoder layer (1 layer)
        dec_id = -1 if use_last_dec_layer else 0
        prefix = f"transformer.decoder.layers.{dec_id}"
        src_dec = types.SimpleNamespace(
            self_attn=types.SimpleNamespace(
                in_proj_weight=sd[f"{prefix}.self_attn.in_proj_weight"],
                in_proj_bias=sd[f"{prefix}.self_attn.in_proj_bias"],
                out_proj=types.SimpleNamespace(
                    weight=sd[f"{prefix}.self_attn.out_proj.weight"],
                    bias=sd[f"{prefix}.self_attn.out_proj.bias"],
                ),
            ),
            multihead_attn=types.SimpleNamespace(
                in_proj_weight=sd[f"{prefix}.multihead_attn.in_proj_weight"],
                in_proj_bias=sd[f"{prefix}.multihead_attn.in_proj_bias"],
                out_proj=types.SimpleNamespace(
                    weight=sd[f"{prefix}.multihead_attn.out_proj.weight"],
                    bias=sd[f"{prefix}.multihead_attn.out_proj.bias"],
                ),
            ),
            linear1=types.SimpleNamespace(
                weight=sd[f"{prefix}.linear1.weight"],
                bias=sd[f"{prefix}.linear1.bias"],
            ),
            linear2=types.SimpleNamespace(
                weight=sd[f"{prefix}.linear2.weight"],
                bias=sd[f"{prefix}.linear2.bias"],
            ),
            norm1=types.SimpleNamespace(
                weight=sd[f"{prefix}.norm1.weight"],
                bias=sd[f"{prefix}.norm1.bias"],
            ),
            norm2=types.SimpleNamespace(
                weight=sd[f"{prefix}.norm2.weight"],
                bias=sd[f"{prefix}.norm2.bias"],
            ),
            norm3=types.SimpleNamespace(
                weight=sd[f"{prefix}.norm3.weight"],
                bias=sd[f"{prefix}.norm3.bias"],
            ),
        )
        _copy_dec_layer(self.visumodel.DE.decoder.layers[0], src_dec)

        print("[DETR seed] Copied: input_proj + encoder(6) + decoder(1)")

    def forward(self, image, mask, word_id, word_mask, return_aux: bool = False):
        # word ids / masks
        if not torch.is_tensor(word_id):
            word_id = torch.as_tensor(word_id, dtype=torch.long, device=image.device)
        else:
            word_id = word_id.to(image.device).long()

        if not torch.is_tensor(word_mask):
            word_mask = torch.as_tensor(word_mask, dtype=torch.long, device=image.device)
        else:
            word_mask = word_mask.to(image.device).long()

        # HF expects 1=real, 0=pad; key_padding_mask needs True=pad
        attn_mask = word_mask                       # (B,L) with 1=real for BERT
        key_pad   = (word_mask == 0).bool()         # (B,L) True=pad for DE/VLF

        # 1) text encoder
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=attn_mask,
            return_dict=True
        )
        fl = outputs.last_hidden_state
        sentence_feature = getattr(outputs, "pooler_output", None)
        if sentence_feature is None:
            am = attn_mask.unsqueeze(-1).float()
            sentence_feature = (fl * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)

        if not self.tunebert:
            fl = fl.detach(); sentence_feature = sentence_feature.detach()

        # 2) visual encoder  —— PASS HF mask (1=real, 0=pad)
        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)

        # 3) fusion  —— PASS HF mask (1=real, 0=pad)
        x = self.vlmodel(fv, fl, word_mask)

        # 4) feature for head - take [pr] token at index 0
        if x.dim() == 2 and x.size(1) == 256:
            feat = x
        elif x.dim() == 3:
            if x.size(1) == 256:
                feat = x[:, :, 0]   # (B,256,S) -> token 0
            elif x.size(2) == 256:
                feat = x[:, 0, :]   # (B,S,256) -> token 0
            else:
                raise RuntimeError(f"Unexpected fusion output shape {x.shape}")
        else:
            raise RuntimeError(f"Unsupported fusion output shape {x.shape}")

        # 5) box head
        outbox = self.box_head(feat).sigmoid()
        return (outbox, {}) if return_aux else outbox
