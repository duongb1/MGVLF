# data_loader.py
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch.utils.data as data

from transformers import AutoTokenizer
from utils.transforms import letterbox
import random

def filelist(root, file_type):
    return [os.path.join(dp, f)
            for dp, _, files in os.walk(root)
            for f in files if f.endswith(file_type)]

class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=640, transform=None, augment=False,
                 split='train', testmode=False, max_query_len=40, bert_model='bert-base-uncased',
                 splits_dir='./DIOR_RSVG/'):   # <-- thêm tham số
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.splits_dir = splits_dir        # <-- lưu lại
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        self.query_len = max_query_len  # 40

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)

        split_file = os.path.join(self.splits_dir, f'{split}.txt')  # <-- dùng path truyền vào
        with open(split_file, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        Index = [int(idx) for idx in lines]

        count = 0
        annotations = sorted(filelist(anno_path, '.xml'))  # nên sort để ổn định
        for anno_path in annotations:
            root = ET.parse(anno_path).getroot()
            for member in root.findall('object'):
                if count in Index:
                    imageFile = os.path.join(self.images_path, root.find('./filename').text)
                    box = np.array([
                        int(member[2][0].text),
                        int(member[2][1].text),
                        int(member[2][2].text),
                        int(member[2][3].text)
                    ], dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)  # x1 y1 x2 y2
        img = cv2.imread(img_path)
        # NEW: BGR -> RGB để khớp Normalize(mean/std) kiểu ImageNet
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, phrase, bbox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()

        # letterbox trả ảnh size x size + mask pad (H,W)
        img, pad_mask, ratio, dw, dh = letterbox(img, None, self.imsize)

        # ratio có thể là scalar hoặc (rx,ry)
        if isinstance(ratio, (tuple, list, np.ndarray)):
            rx, ry = float(ratio[0]), float(ratio[1])
        else:
            rx = ry = float(ratio)

        # === scale bbox sang hệ ảnh đã letterbox (PIXEL XYXY) ===
        bbox = bbox.astype(np.float32)
        bbox[0], bbox[2] = bbox[0] * rx + dw, bbox[2] * rx + dw
        bbox[1], bbox[3] = bbox[1] * ry + dh, bbox[3] * ry + dh

        # === clamp về [0, imsize-1] để tránh tràn ===
        np.clip(bbox, 0, self.imsize - 1, out=bbox)

        # === mask dạng bool (H,W) ===
        pad_mask = pad_mask.astype(bool)

        # Transform ảnh (ToTensor + Normalize)
        if self.transform is not None:
            img = self.transform(img)

        # Tokenize (HF: 1=real, 0=pad)
        enc = self.tokenizer(
            phrase, max_length=self.query_len, padding="max_length",
            truncation=True, return_tensors=None, add_special_tokens=True
        )
        word_id = np.array(enc["input_ids"], dtype=np.int64)
        word_mask = np.array(enc["attention_mask"], dtype=np.int64)

        if self.testmode:
            return (img, pad_mask, word_id, word_mask,
                    bbox.astype(np.float32),           # xyxy PIXEL sau letterbox
                    np.array([rx, ry], np.float32),    # ratio chuẩn hoá về 2 phần tử
                    np.float32(dw), np.float32(dh),
                    self.images[idx][0], phrase)
        else:
            return img, pad_mask, word_id, word_mask, bbox.astype(np.float32)
