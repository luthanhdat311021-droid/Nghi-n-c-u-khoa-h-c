from __future__ import annotations

import numpy as np


def roi_to_px(roi: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return (
        int(round(x1 * w)), int(round(y1 * h)),
        max(int(round(x2 * w)), int(round(x1 * w)) + 1),
        max(int(round(y2 * h)), int(round(y1 * h)) + 1),
    )


def px_to_roi(box: tuple[float, float, float, float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [
        round(max(0.0, x1 / w), 4), round(max(0.0, y1 / h), 4),
        round(min(1.0, x2 / w), 4), round(min(1.0, y2 / h), 4),
    ]


def crop(frame: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_to_px(roi, w, h)
    return frame[y1:y2, x1:x2]


def overlap_ratio(roi_px: tuple[int, int, int, int], box: tuple[float, float, float, float]) -> float:
    """Tỉ lệ diện tích ROI bị box che."""
    ax1, ay1, ax2, ay2 = roi_px
    bx1, by1, bx2, by2 = box
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return (iw * ih) / area
