"""Lấy mẫu khung hình từ camera RTSP / webcam / file video ở tần số cố định (1–5 fps).

- Nguồn trực tiếp (RTSP, webcam): một luồng đọc liên tục để luôn giữ khung mới nhất
  (tránh trễ do bộ đệm), vòng lặp chính lấy mẫu theo đồng hồ thực.
- File video: dùng thời gian của video (CAP_PROP_POS_MSEC) làm đồng hồ, cho phép
  chạy nhanh hơn thời gian thực khi đánh giá offline.
"""
from __future__ import annotations

import os
import re
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
                 width: int | None = None, height: int | None = None, flip: str | None = None,
                 loop: bool = False, rotate: int | None = None, max_side: int | None = None):
        self.source = source
        # Xoay ảnh (độ, theo chiều kim đồng hồ) — vd điện thoại cầm dọc nhưng app gửi ảnh ngang.
        self.rotate_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate or 0)
        # Thu nhỏ để cạnh dài <= max_side (camera điện thoại 1080p làm quy trình chậm, quạt cần >= 10 fps).
        self.max_side = max_side
        self.loop = loop                 # file video: phát lặp lại (thời gian tiếp tục tăng) — để chạy thử/demo
        # Lật ảnh: "horizontal" (bỏ hiệu ứng gương), "vertical" (camera lắp ngược), "both".
        self.flip_code = {"horizontal": 1, "vertical": 0, "both": -1}.get(flip or "", None)
        self.width, self.height = width, height
        self.fps = fps
        self.is_file = Path(source).exists() and not source.isdigit()
        self.t0 = time.time() if t0 is None else t0
        self.realtime = realtime
        self._latest: tuple[np.ndarray, float] | None = None
        self._new_frame = threading.Event()   # luồng đọc báo có khung mới → vòng chính lấy ngay, không lệch pha
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
        elif "://" in str(src):
            # RTSP qua WiFi: TCP tránh mất gói UDP gây vỡ hình (camera Imou/Dahua).
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            # Camera mạng: luồng treo (điện thoại đổi cài đặt, mất WiFi) → read() trả False sau 5 s
            # thay vì chờ mãi → _reader tự kết nối lại.
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
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
            raise RuntimeError(f"Không mở được nguồn video: {re.sub(r'//[^/@]*@', '//***@', str(self.source))}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        it = self._iter_file() if self.is_file else self._iter_live()
        if self.flip_code is None and self.rotate_code is None and not self.max_side:
            return it
        return (Frame(self._transform(f.image), f.ts, f.index) for f in it)

    def _transform(self, img: np.ndarray) -> np.ndarray:
        if self.max_side and max(img.shape[:2]) > self.max_side:
            s = self.max_side / max(img.shape[:2])
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if self.rotate_code is not None:
            img = cv2.rotate(img, self.rotate_code)
        if self.flip_code is not None:
            img = cv2.flip(img, self.flip_code)
        return img

    def _iter_file(self) -> Iterator[Frame]:
        cap = self._open()
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(native / self.fps)))
        i = idx = 0
        try:
            while True:
                ok = cap.grab()
                if not ok and self.loop and i > 0:
                    cap.release()
                    cap = self._open()
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
                    self._new_frame.set()
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
        last_yield = 0.0
        try:
            while True:
                # Chờ khung mới thay vì ngủ cố định 1/fps: ngủ cố định lệch pha với nhịp camera → khung mới
                # đến ngay sau lúc kiểm tra phải chờ thêm cả chu kỳ (webcam 30 fps chỉ còn ~20 fps).
                if not self._new_frame.wait(timeout=1.0):
                    continue
                self._new_frame.clear()
                if self._latest is None or self._latest[1] == last_ts:
                    continue
                # Giới hạn tần số xử lý theo cấu hình (dung sai 20%: khung camera đến lệch nhịp vài ms).
                if self._latest[1] - last_yield < 0.8 * period:
                    continue
                img, last_ts = self._latest
                last_yield = last_ts
                yield Frame(img.copy(), last_ts, idx)
                idx += 1
        finally:
            self._stop.set()
