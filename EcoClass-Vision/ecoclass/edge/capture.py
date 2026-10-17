"""Lấy mẫu khung hình từ camera RTSP / webcam / file video ở tần số cố định (1–5 fps).

- Nguồn trực tiếp (RTSP, webcam): một luồng đọc liên tục để luôn giữ khung mới nhất
  (tránh trễ do bộ đệm), vòng lặp chính lấy mẫu theo đồng hồ thực.
- File video: dùng thời gian của video (CAP_PROP_POS_MSEC) làm đồng hồ, cho phép
  chạy nhanh hơn thời gian thực khi đánh giá offline.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    ts: float          # giây (epoch với nguồn trực tiếp, thời gian video + t0 với file)
    index: int


class FrameSource:
    def __init__(self, source: str, fps: float = 5.0, t0: float | None = None, realtime: bool = False,
                 width: int | None = None, height: int | None = None, flip: str | None = None):
        self.source = source
        # Lật ảnh: "horizontal" (bỏ hiệu ứng gương), "vertical" (camera lắp ngược), "both".
        self.flip_code = {"horizontal": 1, "vertical": 0, "both": -1}.get(flip or "", None)
        self.width, self.height = width, height
        self.fps = fps
        self.is_file = Path(source).exists() and not source.isdigit()
        self.t0 = time.time() if t0 is None else t0
        self.realtime = realtime
        self._latest: tuple[np.ndarray, float] | None = None
        self._stop = threading.Event()
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        src: str | int = int(self.source) if self.source.isdigit() else self.source
        if self.is_file and not self.source.isascii():
            # OpenCV trên Windows không mở được đường dẫn có dấu tiếng Việt → thử đường dẫn tương đối.
            try:
                rel = os.path.relpath(self.source)
                if rel.isascii():
                    src = rel
            except ValueError:
                pass
        if isinstance(src, int) and os.name == "nt":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)   # MSMF hay lỗi grab với webcam laptop
        else:
            cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            # MJPG cho phép webcam USB chạy độ phân giải cao ở FPS cao (YUY2 thường bị giới hạn ~5–10 fps ở 720p).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if self.width and self.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[capture] webcam {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                  f" @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {self.source}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        it = self._iter_file() if self.is_file else self._iter_live()
        if self.flip_code is None:
            return it
        return (Frame(cv2.flip(f.image, self.flip_code), f.ts, f.index) for f in it)

    def _iter_file(self) -> Iterator[Frame]:
        cap = self._open()
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(native / self.fps)))
        i = idx = 0
        try:
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if i % step == 0:
                    ok, img = cap.retrieve()
                    if not ok:
                        break
                    ts = self.t0 + i / native
                    yield Frame(img, ts, idx)
                    idx += 1
                    if self.realtime:
                        time.sleep(1.0 / self.fps)
                i += 1
        finally:
            cap.release()

    def _reader(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._cap = self._open()
                backoff = 1.0
                while not self._stop.is_set():
                    ok, img = self._cap.read()
                    if not ok:
                        break
                    self._latest = (img, time.time())
            except RuntimeError as e:
                print(f"[capture] {e}; thử lại sau {backoff:.0f}s")
            if self._cap is not None:
                self._cap.release()
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _iter_live(self) -> Iterator[Frame]:
        th = threading.Thread(target=self._reader, daemon=True)
        th.start()
        period = 1.0 / self.fps
        idx = 0
        last_ts = None
        try:
            while True:
                tick = time.time()
                if self._latest is not None and self._latest[1] != last_ts:
                    img, last_ts = self._latest
                    yield Frame(img.copy(), last_ts, idx)
                    idx += 1
                time.sleep(max(0.0, period - (time.time() - tick)))
        finally:
            self._stop.set()
