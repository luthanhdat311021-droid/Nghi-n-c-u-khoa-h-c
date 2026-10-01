import math

import cv2
import numpy as np

from ecoclass.config import DeviceConfig
from ecoclass.edge.devices import build_detector, make_scene
from ecoclass.edge.privacy import blur_boxes
from ecoclass.fsm import OFF, ON

W, H = 320, 240


def room(light_on: bool, fan_angle: float | None = None):
    img = np.full((H, W, 3), (150, 165, 175), np.uint8)
    cv2.rectangle(img, (40, 10), (120, 25), (252, 252, 252) if light_on else (95, 95, 95), -1)
    if fan_angle is not None:
        for b in range(3):
            a = math.radians(fan_angle + 120 * b)
            cv2.line(img, (220, 60), (int(220 + 45 * math.cos(a)), int(60 + 20 * math.sin(a))), (40, 50, 60), 8)
    return img


def run_detector(det, frames):
    out = None
    for i, f in enumerate(frames):
        out = det.update(f, make_scene(f, []), i * 0.2)
    return out


def test_light_on_off():
    cfg = dict(id="l", type="light", roi=[0.1, 0.02, 0.4, 0.12], ref_roi=[0.1, 0.5, 0.4, 0.6])
    assert run_detector(build_detector(DeviceConfig(**cfg)), [room(True)] * 4).state == ON
    assert run_detector(build_detector(DeviceConfig(**cfg)), [room(False)] * 4).state == OFF


def test_fan_spinning_vs_static():
    cfg = dict(id="f", type="fan", roi=[0.55, 0.1, 0.9, 0.45])
    spin = [room(True, 47 * i) for i in range(8)]
    still = [room(True, 10) for _ in range(8)]
    assert run_detector(build_detector(DeviceConfig(**cfg)), spin).state == ON
    assert run_detector(build_detector(DeviceConfig(**cfg)), still).state == OFF


def test_hysteresis_requires_consecutive_frames():
    det = build_detector(DeviceConfig(id="l", type="light", roi=[0.1, 0.02, 0.4, 0.12], ref_roi=[0.1, 0.5, 0.4, 0.6]))
    run_detector(det, [room(False)] * 4)
    r = det.update(room(True), make_scene(room(True), []), 1.0)
    assert r.raw == ON and r.state == OFF        # mới 1 khung → chưa lật
    for t in (1.2, 1.4):
        r = det.update(room(True), make_scene(room(True), []), t)
    assert r.state == ON


def test_occluded_roi_keeps_state():
    det = build_detector(DeviceConfig(id="l", type="light", roi=[0.1, 0.02, 0.4, 0.12], ref_roi=[0.1, 0.5, 0.4, 0.6]))
    run_detector(det, [room(True)] * 4)
    f = room(False)
    r = det.update(f, make_scene(f, [(30, 0, 140, 40)]), 2.0)
    assert r.raw == "UNKNOWN" and r.state == ON


def test_occluded_light_still_accepts_on():
    """Người che một phần đèn không thể tạo ra đèn sáng → đọc ON vẫn được tin."""
    det = build_detector(DeviceConfig(id="l", type="light", roi=[0.1, 0.02, 0.4, 0.12], ref_roi=[0.1, 0.5, 0.4, 0.6]))
    run_detector(det, [room(False)] * 4)
    f = room(True)
    for t in (2.0, 2.2, 2.4):
        r = det.update(f, make_scene(f, [(30, 0, 140, 40)]), t)
    assert r.state == ON


def test_blur_changes_only_box():
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_boxes(img, [(40, 40, 60, 60)], pad=0)
    assert np.array_equal(out[:30], img[:30])
    assert not np.array_equal(out[40:60, 40:60], img[40:60, 40:60])
