"""Ảnh "chồng thời gian" (temporal stack) cho bộ phân lớp trạng thái thiết bị chuyển động (quạt).

Ba khung xám cách nhau ``dt`` giây được căn chỉnh tịnh tiến về khung mới nhất (tương quan pha,
bù rung tay/camera xê dịch) rồi ghép thành ảnh 3 kênh [t-2dt, t-dt, t]:
  * vùng đứng yên → ba kênh bằng nhau → màu xám;
  * cánh quạt quay / người cử động → ba kênh khác nhau → viền màu.
Mạng CNN thông thường (MobileNetV3) học được phân biệt quạt quay với quạt đứng yên trên một ảnh.
Dùng chung cho cả tạo dữ liệu huấn luyện (scripts/build_scene_cls.py) và suy luận (devices.py).
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = (384, 224)    # (rộng, cao) ảnh đầu vào bộ phân lớp toàn khung


def to_gray_small(frame_bgr: np.ndarray, size: tuple[int, int] = SIZE) -> np.ndarray:
    g = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA)


def _align(src: np.ndarray, dst: np.ndarray, win: np.ndarray) -> np.ndarray:
    (dx, dy), _ = cv2.phaseCorrelate(src.astype(np.float32), dst.astype(np.float32), win)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(src, m, (src.shape[1], src.shape[0]), borderMode=cv2.BORDER_REPLICATE)


_WIN: dict[tuple[int, int], np.ndarray] = {}


def make_tstack(g2: np.ndarray, g1: np.ndarray, g0: np.ndarray, person_mask: np.ndarray | None = None) -> np.ndarray:
    """g2 = t-2dt, g1 = t-dt, g0 = t (ảnh xám cùng cỡ). Trả về ảnh 3 kênh uint8 (thứ tự kênh B,G,R).

    ``person_mask`` (bool, cỡ bất kỳ): vùng người được xoá tín hiệu chuyển động (ba kênh = khung hiện tại)
    → người đi lại không bị nhầm thành quạt quay."""
    key = (g0.shape[1], g0.shape[0])
    if key not in _WIN:
        _WIN[key] = cv2.createHanningWindow(key, cv2.CV_32F)
    w = _WIN[key]
    st = np.dstack([_align(g2, g0, w), _align(g1, g0, w), g0])
    if person_mask is not None and person_mask.any():
        m = cv2.resize(person_mask.astype(np.uint8), key, interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, np.ones((7, 7), np.uint8)).astype(bool)
        st[m] = g0[m][:, None]
    return st


def person_mask_from_result(r, shape: tuple[int, int]) -> np.ndarray | None:
    """Mặt nạ người (bool HxW) từ kết quả ultralytics YOLO-seg (lớp 0 = person)."""
    if getattr(r, "masks", None) is None or r.boxes is None or not len(r.boxes):
        return None
    mask = np.zeros(shape, np.uint8)
    for poly, c in zip(r.masks.xy, r.boxes.cls.cpu().numpy().astype(int)):
        if c == 0 and len(poly) >= 3:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask.astype(bool)
