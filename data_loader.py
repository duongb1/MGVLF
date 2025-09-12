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
        h, w = img.shape[:2]

        # NHẬN mask bool 2D từ letterbox
        img, pad_mask, ratio, dw, dh = letterbox(img, None, self.imsize)
        
        # Fix ratio handling - check if ratio is tuple/list
        if isinstance(ratio, (tuple, list, np.ndarray)):
            rx, ry = float(ratio[0]), float(ratio[1])
        else:
            rx = ry = float(ratio)
        
        # Scale bbox coordinates
        bbox[0], bbox[2] = bbox[0] * rx + dw, bbox[2] * rx + dw
        bbox[1], bbox[3] = bbox[1] * ry + dh, bbox[3] * ry + dh

        if self.transform is not None:
            img = self.transform(img)  # ToTensor + Normalize

        enc = self.tokenizer(
            phrase, max_length=self.query_len, padding="max_length",
            truncation=True, return_tensors=None, add_special_tokens=True
        )
        word_id = np.array(enc["input_ids"], dtype=int)
        word_mask = np.array(enc["attention_mask"], dtype=int)

        if self.testmode:
            return (img, pad_mask, word_id, word_mask,
                    np.array(bbox, np.float32),
                    np.array(ratio, np.float32),
                    np.array(dw, np.float32),
                    np.array(dh, np.float32),
                    self.images[idx][0], phrase)
        else:
            return img, pad_mask, word_id, word_mask, np.array(bbox, np.float32)
