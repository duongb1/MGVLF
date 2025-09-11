# utils/transforms.py

"""
Generic Image Transform utilities.
"""

import cv2
import random
import math
import numpy as np
from collections.abc import Iterable

import torch
import torch.nn.functional as F


class ResizePad:
    """
    Resize and pad an image to given size.
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError(f'Got inappropriate size arg: {size}')
        self.h, self.w = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        resized_img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        if img.ndim > 2:
            new_img = np.zeros((self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)
        else:
            resized_img = np.expand_dims(resized_img, -1)
            new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)

        new_img[pad_h: pad_h + resized_h, pad_w: pad_w + resized_w, ...] = resized_img
        return new_img


class CropResize:
    """Remove padding and resize image to its original size."""

    def __call__(self, img: torch.Tensor, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError(f'Got inappropriate size arg: {size}')

        im_h, im_w = img.shape[:2] if img.ndim == 2 else img.shape[-2:]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        crop_h = int(np.floor(resized_h - input_h) / 2)
        crop_w = int(np.floor(resized_w - input_w) / 2)

        if img.ndim == 2:
            # (H, W) -> (1,1,H,W)
            resized_img = F.interpolate(
                img.unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # -> (H, W)
            return resized_img[crop_h: crop_h + input_h, crop_w: crop_w + input_w]
        else:
            # (C, H, W) -> (1,C,H,W)
            resized_img = F.interpolate(
                img.unsqueeze(0),
                size=(resized_h, resized_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # -> (C, H, W)
            return resized_img[:, crop_h: crop_h + input_h, crop_w: crop_w + input_w]


class ResizeImage:
    """Resize the largest of the sides of the image to a given size"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError(f'Got inappropriate size arg: {size}')
        self.size = size

    def __call__(self, img: torch.Tensor):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.interpolate(
            img.unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return out


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError(f'Got inappropriate size arg: {size}')
        self.size = size

    def __call__(self, img: torch.Tensor):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=(resized_h, resized_w),
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        return out


class ToNumpy:
    """Transform a torch.*Tensor to a numpy ndarray."""
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()


def letterbox(img: np.ndarray, mask, height: int, color=(123.7, 116.3, 103.5)):
    """
    Resize a rectangular image to a padded square of size (height, height).
    Returns:
        img_padded: np.ndarray (H, W, 3)
        mask_bool:  np.ndarray (H, W) bool  (True = padding)
        ratio:      float
        dw, dh:     float (half padding width/height on each side before rounding)
    """
    shape = img.shape[:2]  # (H, W)
    ratio = float(height) / max(shape)
    new_w, new_h = round(shape[1] * ratio), round(shape[0] * ratio)
    dw = (height - new_w) / 2
    dh = (height - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resize + pad image
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    # build padding mask (True where padded)
    base = np.zeros((new_h, new_w), dtype=np.uint8)
    pad_mask = cv2.copyMakeBorder(
        base, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=1  # 1 in padded region
    )
    mask_bool = pad_mask.astype(bool)

    return img, mask_bool, ratio, dw, dh


def random_affine(img, mask, targets, degrees=(-10, 10), translate=(.1, .1),
                  scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5), all_bbox=None):
    """
    Apply random affine transform to image (and optionally mask/bboxes).
    - img: np.ndarray HxWx3
    - mask: np.ndarray HxW (bool or uint8) or None
    - targets: bbox [x1,y1,x2,y2] or list of such bboxes or None
    Returns variants depending on inputs (kept same as original code).
    """
    border = 0
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

    M = S @ T @ R  # Combined rotation matrix

    imw = cv2.warpPerspective(img, M, dsize=(height, height),
                              flags=cv2.INTER_LINEAR, borderValue=borderValue)

    if mask is not None:
        # Handle bool/uint8 masks consistently
        if mask.dtype == np.bool_:
            mask_u8 = mask.astype(np.uint8) * 1
        else:
            mask_u8 = mask.astype(np.uint8)
        maskw = cv2.warpPerspective(mask_u8, M, dsize=(height, height),
                                    flags=cv2.INTER_NEAREST, borderValue=1)
        maskw = (maskw > 0)
    else:
        maskw = None

    # Return warped boxes as in original API
    if isinstance(targets, list):
        targetlist = []
        for bbox in targets:
            targetlist.append(wrap_points(bbox, M, height, a))
        return imw, maskw, targetlist, M
    elif all_bbox is not None:
        targets = wrap_points(targets, M, height, a)
        for ii in range(all_bbox.shape[0]):
            all_bbox[ii, :] = wrap_points(all_bbox[ii, :], M, height, a)
        return imw, maskw, targets, all_bbox, M
    elif targets is not None:
        targets = wrap_points(targets, M, height, a)
        return imw, maskw, targets, M
    else:
        return imw


def wrap_points(targets, M, height, a):
    """
    Warp a single bbox [x1, y1, x2, y2] with homography M and clamp/validate.
    """
    points = targets.copy()
    area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x_c = (xy[:, 2] + xy[:, 0]) / 2
    y_c = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2)).reshape(4, 1).T

    # clamp to image
    np.clip(xy, 0, height, out=xy)
    return xy[0]
