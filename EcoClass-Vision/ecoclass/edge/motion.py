"""Ước lượng chuyển động camera (tịnh tiến) để ROI bám theo thiết bị khi camera bị xê dịch.

Dùng tương quan pha (cv2.phaseCorrelate) giữa khung hiện tại và **khung tham chiếu** (khung lúc
cấu hình/chọn ROI), không cộng dồn từng khung → không bị trôi tích luỹ. Điểm ảnh thuộc người
được thay bằng giá trị trung vị để người cử động không kéo lệch kết quả. Khi độ tin cậy
(response) thấp — camera quay sang hướng khác, khung không còn chồng lấp với tham chiếu —
``reliable = False`` và các thiết bị bám ROI báo "ngoài khung" (UNKNOWN) thay vì đo nhầm chỗ.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, width: int = 320, moving_px: float = 3.0, min_response: float = 0.08,
                 max_shift: float = 0.45):
        self.width = width
        self.moving_px = moving_px          # đổi độ dịch giữa hai khung lớn hơn → camera đang di chuyển
        self.min_response = min_response
        self.max_shift = max_shift          # dịch quá tỉ lệ này của khung → coi như mất tham chiếu
        self.ref: np.ndarray | None = None
        self.win: np.ndarray | None = None
        self.ox = self.oy = 0.0
        self.moving = False
        self.reliable = True
        self.response = 1.0

    def _prep(self, gray, person_mask):
        h, w = gray.shape[:2]
        s = self.width / w
        g = cv2.resize(gray, (self.width, int(h * s)), interpolation=cv2.INTER_AREA).astype(np.float32)
        if person_mask is not None:
            m = cv2.resize(person_mask.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if m.any() and (~m).any():
                g[m] = float(np.median(g[~m]))
        return g, s

    def update(self, gray: np.ndarray, person_mask: np.ndarray | None = None) -> tuple[float, float, bool]:
        g, s = self._prep(gray, person_mask)
        if self.ref is None or self.ref.shape != g.shape:
            self.ref = g
            self.win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            self.ox = self.oy = 0.0
            self.reliable, self.moving = True, False
            return self.ox, self.oy, self.moving
        (dx, dy), resp = cv2.phaseCorrelate(self.ref, g, self.win)
        self.response = float(resp)
        dx, dy = dx / s, dy / s
        H, W = gray.shape[:2]
        ok = resp >= self.min_response and abs(dx) < self.max_shift * W and abs(dy) < self.max_shift * H
        if ok:
            self.moving = bool(np.hypot(dx - self.ox, dy - self.oy) > self.moving_px) or not self.reliable
            self.ox, self.oy = float(dx), float(dy)
        else:
            self.moving = True
        self.reliable = bool(ok)
        return self.ox, self.oy, self.moving
