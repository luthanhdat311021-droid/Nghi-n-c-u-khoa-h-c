"""Đọc cấu hình phòng học (YAML) thành các dataclass có kiểu rõ ràng.

Toạ độ ROI luôn lưu ở dạng chuẩn hoá [x1, y1, x2, y2] trong khoảng 0..1 để
không phụ thuộc độ phân giải camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PersonConfig:
    model: str = "yolo11n.pt"
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    # Ngưỡng tin cậy có trễ: khi phòng đang "có người" thì hạ ngưỡng để không
    # đánh mất người ngồi yên / bị che khuất; khi phòng trống thì nâng ngưỡng
    # để tránh báo có người giả.
    conf_occupied: float = 0.25
    conf_empty: float = 0.40
    hold_seconds: float = 3.0      # giữ ID đã theo vết thêm một lúc khi bị che
    window: int = 5                # cửa sổ trượt làm trơn N(t)
    device: str | None = None      # "cpu", "0", ... (None = tự chọn)
    half: bool = False             # FP16 trên GPU NVIDIA → nhanh hơn


@dataclass
class FSMConfig:
    empty_seconds_to_pending: float = 2.0   # N=0 liên tục bao lâu thì "nghi vắng"
    te_seconds: float = 300.0               # Te khi phòng có lịch học
    te_no_class_seconds: float = 120.0      # Te rút ngắn khi không có lịch
    off_confirm_seconds: float = 5.0        # thiết bị TẮT ổn định bao lâu thì "đã xử lý"


@dataclass
class DeviceConfig:
    id: str
    type: str                       # light | fan | screen | ac | monitor
    roi: list[float]                # [x1, y1, x2, y2] chuẩn hoá
    power_w: float = 0.0
    ref_roi: list[float] | None = None   # vùng nền để tính tỉ số độ sáng
    classifier: str | None = None        # đường dẫn MobileNetV3 (.onnx/.pt), tuỳ chọn
    params: dict[str, Any] = field(default_factory=dict)
    on_frames: int = 3              # số lần đọc liên tiếp để lật trạng thái
    off_frames: int = 3


@dataclass
class PublisherConfig:
    mode: str = "http"              # http | mqtt | stdout
    http_url: str = "http://127.0.0.1:8000"
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    topic_prefix: str = "ecoclass"
    heartbeat_seconds: float = 10.0
    api_key: str | None = None


@dataclass
class RoomConfig:
    room_id: str
    source: str
    fps: float = 5.0
    capture_width: int | None = None     # yêu cầu độ phân giải webcam (vd 1280x720)
    capture_height: int | None = None
    timetable: str | None = None
    tariff_vnd_per_kwh: float = 2500.0
    person: PersonConfig = field(default_factory=PersonConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    devices: list[DeviceConfig] = field(default_factory=list)
    publisher: PublisherConfig = field(default_factory=PublisherConfig)
    evidence_max_width: int = 960
    base_dir: str = "."             # thư mục chứa file YAML (để giải đường dẫn tương đối)

    def resolve(self, p: str | None) -> str | None:
        """Giải đường dẫn tương đối theo vị trí file cấu hình."""
        if p is None:
            return None
        if "://" in p or p.isdigit():
            return p
        path = Path(p)
        if path.is_absolute():
            return str(path)
        cand = Path(self.base_dir) / path
        return str(cand) if cand.exists() else p

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("base_dir", None)
        return d


def load_room_config(path: str | Path) -> RoomConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = RoomConfig(
        room_id=raw["room_id"],
        source=str(raw["source"]),
        fps=float(raw.get("fps", 5.0)),
        capture_width=raw.get("capture_width"),
        capture_height=raw.get("capture_height"),
        timetable=raw.get("timetable"),
        tariff_vnd_per_kwh=float(raw.get("tariff_vnd_per_kwh", 2500.0)),
        person=PersonConfig(**raw.get("person", {})),
        fsm=FSMConfig(**raw.get("fsm", {})),
        devices=[DeviceConfig(**d) for d in raw.get("devices", [])],
        publisher=PublisherConfig(**raw.get("publisher", {})),
        evidence_max_width=int(raw.get("evidence_max_width", 960)),
        base_dir=str(path.parent.resolve()),
    )
    ids = [d.id for d in cfg.devices]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Trùng id thiết bị trong {path}: {ids}")
    for d in cfg.devices:
        _check_roi(d.roi, d.id)
        if d.ref_roi is not None:
            _check_roi(d.ref_roi, d.id + ".ref_roi")
    return cfg


def save_room_config(cfg: RoomConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _check_roi(roi: list[float], name: str) -> None:
    if len(roi) != 4:
        raise ValueError(f"ROI của {name} phải có 4 giá trị")
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"ROI của {name} không hợp lệ (cần 0<=x1<x2<=1, 0<=y1<y2<=1): {roi}")
