# models/model.py
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CNN_MGVLF import build_VLFusion, build_CNN_MGVLF
from transformers import AutoModel, AutoConfig


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
