"""Auto-ROI: định vị thiết bị trong phòng bằng YOLO-World theo câu lệnh văn bản.

Chạy một lần khi lắp đặt camera; kết quả ghi vào file cấu hình phòng để kỹ thuật
viên rà soát. Không cần gán nhãn thủ công khi nhân rộng sang phòng mới.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import px_to_roi

# Câu lệnh văn bản -> loại thiết bị của hệ thống
DEFAULT_PROMPTS: dict[str, str] = {
    "ceiling light": "light",
    "fluorescent lamp": "light",
    "ceiling fan": "fan",
    "wall fan": "fan",
    "projector screen": "screen",
    "projector": "projector",
    "air conditioner": "ac",
    "computer monitor": "monitor",
    "television": "tv",
}

DEFAULT_POWER_W = {"light": 36, "fan": 60, "screen": 250, "projector": 250, "ac": 1000, "monitor": 30, "tv": 120}


@dataclass
class ROICandidate:
    type: str
    prompt: str
    conf: float
    roi: list[float]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_rois(
    frames: list[np.ndarray],
    model_path: str = "yolov8s-worldv2.pt",
    prompts: dict[str, str] | None = None,
    conf: float = 0.08,
    imgsz: int = 960,
    iou_merge: float = 0.5,
    device: str | None = None,
) -> list[ROICandidate]:
    """Phát hiện thiết bị trên vài khung (lấy ở các thời điểm khác nhau), gộp box trùng.

    Box xuất hiện ổn định qua nhiều khung được giữ lại; điểm = trung bình tin cậy.
    """
    from ultralytics import YOLOWorld

    prompts = prompts or DEFAULT_PROMPTS
    names = list(prompts)
    model = YOLOWorld(model_path)
    model.set_classes(names)

    clusters: list[dict] = []
    for img in frames:
        h, w = img.shape[:2]
        kw = dict(conf=conf, imgsz=imgsz, verbose=False)
        if device is not None:
            kw["device"] = device
        r = model.predict(img, **kw)[0]
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            prompt = names[int(k)]
            box = px_to_roi(tuple(b), w, h)
            for cl in clusters:
                if prompts[cl["prompt"]] == prompts[prompt] and _iou(cl["roi"], box) >= iou_merge:
                    cl["confs"].append(float(c))
                    n = len(cl["confs"])
                    cl["roi"] = [round((cl["roi"][i] * (n - 1) + box[i]) / n, 4) for i in range(4)]
                    break
            else:
                clusters.append({"prompt": prompt, "roi": box, "confs": [float(c)]})

    min_hits = max(1, len(frames) // 2)
    out = [
        ROICandidate(type=prompts[cl["prompt"]], prompt=cl["prompt"], conf=round(float(np.mean(cl["confs"])), 3), roi=cl["roi"])
        for cl in clusters if len(cl["confs"]) >= min_hits
    ]
    return sorted(out, key=lambda c: (c.type, -c.conf))


def suggest_ref_roi(roi: list[float], margin: float = 0.06) -> list[float]:
    """Vùng nền tham chiếu cạnh ROI (ưu tiên bên dưới — tường/trần gần thiết bị)."""
    x1, y1, x2, y2 = roi
    h = y2 - y1
    ny1 = min(1.0 - 0.02, y2 + margin * 0.5)
    ny2 = min(1.0, ny1 + max(0.03, h * 0.6))
    if ny2 - ny1 < 0.02:  # sát mép dưới → lấy phía trên
        ny2 = max(0.02, y1 - margin * 0.5)
        ny1 = max(0.0, ny2 - max(0.03, h * 0.6))
    return [round(x1, 4), round(ny1, 4), round(x2, 4), round(ny2, 4)]
