# models/model.py
from collections import OrderedDict
from typing import Optional

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
    Implementation đúng theo pipeline của paper RSVG/MGVLF:

      Text encoder (BERT, lấy word-level + 1 sentence) →
      CNN_MGVLF (DE: refine fv1 bằng [multi-scale vis + word + sentence]) →
      VLFusion (encoder với [word-level tokens + 1 learnable token]) →
      Box head (cx,cy,w,h) ∈ [0,1]

    Lưu ý: Không dùng WLP/FiLM/Quality head để khớp paper gốc.
    """
    def __init__(self, bert_model: str = 'bert-base-uncased', tunebert: bool = True, args: Optional[object] = None):
        super(MGVLF, self).__init__()
        self.tunebert = tunebert
        self.args = args

        # -------- Text model (HF Transformers) --------
        # output_hidden_states không bắt buộc khi dùng last_hidden_state,
        # vẫn bật để tương thích nếu cần debug.
        config = AutoConfig.from_pretrained(
            bert_model,
            output_hidden_states=True,
            add_pooling_layer=True,  # cố gắng có pooler_output nếu checkpoint hỗ trợ
        )
        self.textmodel = AutoModel.from_pretrained(bert_model, config=config)

        # Nếu không fine-tune BERT, freeze toàn bộ tham số để tiết kiệm compute
        if not self.tunebert:
            for p in self.textmodel.parameters():
                p.requires_grad = False

        # -------- Visual model (CNN branch của MGVLF) --------
        self.visumodel = build_CNN_MGVLF(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.visumodel = load_weights(self.visumodel, args.pretrain)

        # -------- Multimodal Fusion model --------
        self.vlmodel = build_VLFusion(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.vlmodel = load_weights(self.vlmodel, args.pretrain)

        # ===== SEED TỪ DETR: KHÔNG DÙNG PARSER, GỌI THẲNG Ở ĐÂY =====
        try:
            import os
            ckpt = "./pretrained/detr_resnet50.pth"  # nếu bạn có file local, đặt vào đây
            if os.path.isfile(ckpt):
                self.seed_from_detr(ckpt_path=ckpt)          # seed từ file local
            else:
                self.seed_from_detr()                         # seed từ torch.hub (online lần đầu)
            print("[DETR seed] done.")
        except Exception as e:
            print(f"[DETR seed] skipped: {e}")

        # -------- Localization Head --------
        self.box_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        # Alias giữ tương thích nếu ở nơi khác tham chiếu Prediction_Head
        self.Prediction_Head = self.box_head

        for m in self.box_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def seed_from_detr(self, ckpt_path: str | None = None, use_last_dec_layer: bool = False):
        """
        Khởi tạo các phần tương thích từ DETR-ResNet50 pretrained:
        - input_proj 2048->256
        - 6 encoder layers (cho cả encoder ảnh & encoder fusion)
        - 1 decoder layer (cho DE của nhánh ảnh)
        Nếu cung cấp ckpt_path: sẽ load state_dict vào model DETR hub rồi copy (offline-friendly).
        """
        # 1) Tạo model DETR hub (cần internet lần đầu để fetch kiến trúc)
        detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True if ckpt_path is None else False)
        detr.eval()

        # 2) Nếu có ckpt riêng -> nạp vào detr
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location='cpu')
            sd = sd.get('model', sd.get('state_dict', sd))
            detr.load_state_dict(sd, strict=False)

        # 3) input_proj
        with torch.no_grad():
            self.visumodel.input_proj.weight.copy_(detr.input_proj.weight)
            self.visumodel.input_proj.bias.copy_(detr.input_proj.bias)

        # 4) Encoder: copy 6 layers cho EN ảnh và EN fusion
        for dst, src in zip(self.visumodel.transformer.encoder.layers,
                            detr.transformer.encoder.layers):
            _copy_enc_layer(dst, src)

        for dst, src in zip(self.vlmodel.transformer.encoder.layers,
                            detr.transformer.encoder.layers):
            _copy_enc_layer(dst, src)

        # 5) Decoder: model bạn dùng 1 layer -> copy từ layer[0] (hoặc [-1] nếu muốn)
        dec_src = detr.transformer.decoder.layers[-1] if use_last_dec_layer else detr.transformer.decoder.layers[0]
        _copy_dec_layer(self.visumodel.DE.decoder.layers[0], dec_src)

        print("[DETR seed] Copied: input_proj + encoder(6) + decoder(1)")

    def forward(self, image, mask, word_id, word_mask, return_aux: bool = False):
        """
        Inputs:
            image:     (B, 3, H, W) tensor (đã ToTensor + Normalize)
            mask:      (B, H, W) bool, True = padding (chuẩn DETR)
            word_id:   (B, L) input_ids (có thể là np.array)
            word_mask: (B, L) attention_mask (1=token thật, 0=pad; có thể là np.array)
            return_aux: giữ cho tương thích (không dùng trong paper)

        Output:
            outbox: (B, 4) bbox normalized [0,1] in format (cx, cy, w, h)
        """
        # Đưa word_id/word_mask về LongTensor đúng device
        if not torch.is_tensor(word_id):
            word_id = torch.as_tensor(word_id, dtype=torch.long, device=image.device)
        else:
            word_id = word_id.to(image.device).long()

        if not torch.is_tensor(word_mask):
            word_mask = torch.as_tensor(word_mask, dtype=torch.long, device=image.device)
        else:
            word_mask = word_mask.to(image.device).long()

        # -------- 1) Language encoder --------
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            return_dict=True
        )
        # Word-level tokens
        fl = outputs.last_hidden_state  # (B, L, H=768)

        # Sentence-level feature: ưu tiên pooler_output, fallback mean-pool theo mask
        sentence_feature = getattr(outputs, "pooler_output", None)
        if sentence_feature is None:
            am = word_mask.unsqueeze(-1).float()             # (B, L, 1)
            sentence_feature = (fl * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)  # (B, 768)

        if not self.tunebert:
            # nếu đã tắt grad, detach cho chắc (tránh backprop qua BERT)
            fl = fl.detach()
            sentence_feature = sentence_feature.detach()

        # -------- 2) Visual encoder (CNN_MGVLF branch) --------
        # mask ảnh ở đây là True=padding (đúng quy ước DETR)
        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)  # (B, 256, H', W')

        # -------- 3) Fusion encoder --------
        # VLFusion: chỉ ghép word-level + 1 learnable token (pr), không đưa sentence vào fusion.
        x = self.vlmodel(fv, fl, word_mask)  # thường trả (B, 256)

        # -------- 4) Lấy đặc trưng cho head --------
        if x.dim() == 2 and x.size(1) == 256:
            feat = x  # (B,256)
        elif x.dim() == 3:
            # mềm dẻo nếu transformer trả (B,256,L) hoặc (B,L,256)
            if x.size(1) == 256:      # (B,256,L) → lấy token cuối
                feat = x[:, :, -1]
            elif x.size(2) == 256:    # (B,L,256) → lấy token cuối
                feat = x[:, -1, :]
            else:
                raise RuntimeError(f"Unexpected fusion output shape {x.shape}")
        else:
            raise RuntimeError(f"Unsupported fusion output shape {x.shape}")

        # -------- 5) Head --------
        outbox = self.box_head(feat).sigmoid()  # (B,4) in [0,1]

        # paper không dùng aux; giữ cờ để tương thích
        if return_aux:
            return outbox, {}
        return outbox
