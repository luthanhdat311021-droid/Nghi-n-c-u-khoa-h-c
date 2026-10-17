"""Bộ phát hiện thiết bị (YOLO11 huấn luyện trên dữ liệu phòng học: lớp light, fan).

Vai trò: *định vị* máng đèn và quạt trong khung. Trạng thái BẬT/TẮT vẫn do đặc trưng vật lý
quyết định nhưng chỉ đo **bên trong** các box này (LightGroupDetector, FanAutoDetector) →
cửa sổ, tường cháy sáng, người cử động ngoài box quạt không còn gây báo sai.
"""
from __future__ import annotations

import numpy as np


class DeviceLocator:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640, device: str | None = None,
                 max_area: dict[str, float] | None = None):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        # Box quá lớn so với khung thường là phát hiện sai (vd. cả trần nhà bị coi là "quạt").
        self.max_area = {"fan": 0.35, **(max_area or {})}
        self.names = {int(k): v for k, v in self.model.names.items()}

    def __call__(self, frame: np.ndarray) -> dict[str, list[list[float]]]:
        kw = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        r = self.model.predict(frame, **kw)[0]
        out: dict[str, list[list[float]]] = {v: [] for v in self.names.values()}
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                name = self.names[c]
                if (b[2] - b[0]) * (b[3] - b[1]) > self.max_area.get(name, 1.0):
                    continue
                out[name].append([round(float(v), 4) for v in b])
        return out
