"""Phát hiện và đếm người: YOLO11 (tinh chỉnh) + ByteTrack, làm trơn N(t) theo cửa sổ trượt.

Kỹ thuật chống bỏ sót người ngồi yên/bị che:
  * Ngưỡng tin cậy có trễ (conf_occupied < conf_empty).
  * ByteTrack ghép cả box điểm thấp vào vết đã có → một vết đã "xác nhận" (từng
    vượt ngưỡng) vẫn được đếm khi điểm tin cậy tạm thời giảm.
  * Giữ ID trong hold_seconds sau lần thấy cuối.
  * N(t) = trung vị của cửa sổ trượt các giá trị đếm thô.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import PersonConfig

PERSON_CLASS = 0


@dataclass
class PersonResult:
    n_raw: int
    n: int
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    mask: np.ndarray | None = None      # bool HxW (chỉ khi dùng mô hình -seg)


class PersonCounter:
    def __init__(self, cfg: PersonConfig, use_tracker: bool = True, model_path: str | None = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.use_tracker = use_tracker
        self.model = YOLO(model_path or cfg.model)
        self.window: deque[int] = deque(maxlen=max(1, cfg.window))
        self.confirmed: set[int] = set()
        self.last_seen: dict[int, float] = {}
        names = getattr(self.model, "names", {0: "person"})
        self.person_cls = next((k for k, v in names.items() if v == "person"), PERSON_CLASS) if isinstance(names, dict) else PERSON_CLASS
        self.detect_stride = getattr(cfg, "detect_stride", 1)
        self._frame_count = 0
        self._last_result: PersonResult | None = None

    def update(self, frame: np.ndarray, ts: float, occupied: bool) -> PersonResult:
        if self.detect_stride > 1 and self._last_result is not None:
            if self._frame_count % self.detect_stride != 0:
                self._frame_count += 1
                return self._last_result
        self._frame_count += 1

        thr = self.cfg.conf_occupied if occupied else self.cfg.conf_empty
        kw = dict(imgsz=self.cfg.imgsz, classes=[self.person_cls], verbose=False)
        if self.cfg.device is not None:
            kw["device"] = self.cfg.device

        if self.use_tracker:
            # Hạ conf của bộ phát hiện để ByteTrack nhận cả box điểm thấp (liên kết lần 2).
            r = self.model.track(frame, persist=True, tracker=self.cfg.tracker, conf=0.1, **kw)[0]
        else:
            r = self.model.predict(frame, conf=thr, **kw)[0]

        boxes, confs, ids, kept = [], [], [], []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            tid = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(cf)
            for j, (b, c, i) in enumerate(zip(xyxy, cf, tid)):
                if self.use_tracker and i >= 0:
                    if c >= thr:
                        self.confirmed.add(int(i))
                    if int(i) not in self.confirmed:
                        continue
                    self.last_seen[int(i)] = ts
                elif c < thr:
                    continue
                boxes.append(tuple(float(v) for v in b))
                confs.append(float(c))
                ids.append(int(i))
                kept.append(j)

        # Mô hình phân đoạn (yolo11*-seg): mặt nạ người chính xác hơn box để tính che khuất.
        mask = None
        if getattr(r, "masks", None) is not None and kept:
            mask = np.zeros(frame.shape[:2], np.uint8)
            for j in kept:
                poly = r.masks.xy[j]
                if len(poly) >= 3:
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8)).astype(bool)

        if self.use_tracker:
            # Đếm cả vết vừa mất dấu trong hold_seconds (bị che tạm thời).
            self.last_seen = {i: t for i, t in self.last_seen.items() if ts - t <= self.cfg.hold_seconds}
            n_raw = max(len(boxes), len(self.last_seen))
        else:
            n_raw = len(boxes)

        self.window.append(n_raw)
        n = int(round(float(np.median(self.window))))
        res = PersonResult(n_raw=n_raw, n=n, boxes=boxes, confs=confs, ids=ids, mask=mask)
        self._last_result = res
        return res
