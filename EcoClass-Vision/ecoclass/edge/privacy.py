"""Riêng tư theo thiết kế: làm mờ vùng người trước khi ảnh rời khỏi thiết bị biên."""
from __future__ import annotations

import cv2
import numpy as np


def blur_boxes(img: np.ndarray, boxes, pad: float = 0.15, pixel: int = 12) -> np.ndarray:
    """Pixel hoá + làm mờ Gauss các box (x1,y1,x2,y2 theo pixel). Trả về ảnh mới."""
    out = img.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - pad * bw)); y1 = int(max(0, y1 - pad * bh))
        x2 = int(min(w, x2 + pad * bw)); y2 = int(min(h, y2 + pad * bh))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = out[y1:y2, x1:x2]
        small = cv2.resize(region, (max(1, (x2 - x1) // pixel), max(1, (y2 - y1) // pixel)),
                           interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (0, 0), sigmaX=max(3, pixel // 2))
    return out


def blur_mask(img: np.ndarray, mask: np.ndarray, pixel: int = 14) -> np.ndarray:
    """Làm mờ đúng theo hình dáng người (mặt nạ phân đoạn) — giữ nguyên phần nền xung quanh."""
    out = img.copy()
    if mask is None or not mask.any():
        return out
    # Chỉ xử lý trong khung bao của mặt nạ, toàn phép uint8 → ~vài ms ở 720p.
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
    sub = img[y:y + bh, x:x + bw]
    small = cv2.resize(sub, (max(1, bw // pixel), max(1, bh // pixel)), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR)
    m = mask[y:y + bh, x:x + bw]
    out[y:y + bh, x:x + bw][m] = mosaic[m]
    return out


def encode_jpeg(img: np.ndarray, max_width: int = 960, quality: int = 80) -> bytes:
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Không mã hoá được JPEG")
    return buf.tobytes()
