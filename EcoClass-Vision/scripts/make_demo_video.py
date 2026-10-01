"""Sinh video phòng học tổng hợp để kiểm thử đầu-cuối khi chưa có camera thật.

Cảnh: 2 dãy đèn trần, 1 quạt trần 3 cánh, màn chiếu, bàn ghế, cửa sổ (ánh sáng tự nhiên),
người thật được cắt từ ảnh mẫu của Ultralytics (bus.jpg) để YOLO nhận dạng được.

Kịch bản (giây):
    0–15   3 sinh viên trong phòng; đèn, quạt bật; máy chiếu bật đến giây 12
    15     sinh viên ra về, quên tắt đèn và quạt          ← bắt đầu "bỏ quên"
    50     đèn được tắt từ công tắc hành lang
    58     quạt được tắt, quay chậm dần đến giây 61      ← kết thúc "bỏ quên"
    75–85  bảo vệ vào kiểm tra rồi ra (thiết bị đều tắt)
    110    hết video

Đầu ra: video .mp4 và file nhãn sự kiện thật (ground truth) .csv cho scripts/evaluate_events.py.

    python scripts/make_demo_video.py --out data/demo
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
LIGHTS = [(290, 18, 490, 50), (790, 18, 990, 50)]
FAN_C, FAN_R = (640, 115), 95
SCREEN = (900, 140, 1220, 330)
WINDOW = (30, 120, 200, 360)


def person_crops() -> list[np.ndarray]:
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    r = YOLO("yolo11n.pt").predict(img, classes=[0], conf=0.5, verbose=False)[0]
    crops = []
    for b in r.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = b
        if (y2 - y1) > 300 and x1 > 5:  # bỏ người bị cắt ở mép ảnh
            crops.append(img[y1:y2, x1:x2].copy())
    if not crops:
        raise RuntimeError("Không cắt được người mẫu từ bus.jpg")
    return crops


def base_scene() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (150, 170, 178)                       # tường (BGR)
    img[:60] = (120, 128, 132)                     # trần
    img[470:] = (95, 110, 120)                     # sàn
    cv2.rectangle(img, (W // 2 - 4, 60), (W // 2 + 4, FAN_C[1]), (60, 60, 60), -1)  # cần quạt
    for row, y in enumerate((500, 590)):           # bàn
        for x in range(260, 1180, 230):
            cv2.rectangle(img, (x + row * 30, y), (x + 170 + row * 30, y + 40), (40, 75, 120), -1)
            cv2.rectangle(img, (x + 15 + row * 30, y + 40), (x + 25 + row * 30, y + 95), (30, 50, 80), -1)
    return img


def draw_lights(img, on: bool, factor: float):
    for x1, y1, x2, y2 in LIGHTS:
        if on:
            glow = np.zeros_like(img)
            cv2.rectangle(glow, (x1 - 25, y1 - 10), (x2 + 25, y2 + 40), (255, 255, 255), -1)
            glow = cv2.GaussianBlur(glow, (0, 0), 18)
            img[:] = cv2.add(img, (glow * 0.45).astype(np.uint8))
            cv2.rectangle(img, (x1, y1), (x2, y2), (252, 253, 255), -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c * factor) for c in (165, 170, 172)), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 2)


def draw_fan(img, angle: float, speed: float):
    cx, cy = FAN_C
    n_sub = 1 + int(min(speed, 60) / 6)           # nhòe chuyển động khi quay nhanh
    acc = np.zeros(img.shape, np.float32)
    for k in range(n_sub):
        layer = img.copy()
        a0 = angle - k * speed * 0.5 / n_sub
        for b in range(3):
            a = math.radians(a0 + b * 120)
            tip = (int(cx + FAN_R * math.cos(a)), int(cy + 0.35 * FAN_R * math.sin(a)))
            cv2.line(layer, (cx, cy), tip, (45, 60, 70), 16)
        acc += layer
    img[:] = (acc / n_sub).astype(np.uint8)
    cv2.circle(img, (cx, cy), 14, (40, 40, 40), -1)


def draw_screen(img, on: bool, factor: float, t: float):
    x1, y1, x2, y2 = SCREEN
    if on:
        cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 245), -1)
        cv2.putText(img, "Chuong 3: YOLO11", (x1 + 15, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 40, 20), 2)
        for i in range(4):
            cv2.line(img, (x1 + 20, y1 + 65 + 22 * i), (x1 + 200 - 25 * i, y1 + 65 + 22 * i), (60, 60, 60), 3)
        for i, hgt in enumerate((50, 80, 65, 100)):
            cv2.rectangle(img, (x1 + 210 + 24 * i, y2 - 20 - hgt), (x1 + 228 + 24 * i, y2 - 20), (40 + 50 * i, 140, 220 - 40 * i), -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c * factor) for c in (205, 205, 205)), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 2)


def paste(img, crop, cx, bottom, height, sway=0.0):
    s = height / crop.shape[0]
    c = cv2.resize(crop, (int(crop.shape[1] * s), height))
    x1 = int(cx - c.shape[1] / 2 + sway)
    y1 = bottom - height
    x2, y2 = min(W, x1 + c.shape[1]), min(H, y1 + height)
    img[y1:y2, x1:x2] = c[: y2 - y1, : x2 - x1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--duration", type=float, default=110)
    ap.add_argument("--room", default="A1-101")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    people = person_crops()
    base = base_scene()
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(str(out / "demo_classroom.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    angle = 0.0
    n = int(a.duration * a.fps)
    for i in range(n):
        t = i / a.fps
        lights_on = t < 50
        projector_on = t < 12
        if t < 58:
            speed = 23.0                                    # độ/khung (≈ 0.64 vòng/s ở 10 fps)
        elif t < 61:
            speed = 23.0 * (61 - t) / 3                     # quay chậm dần sau khi tắt
        else:
            speed = 0.0
        angle = (angle + speed) % 360
        factor = 1.0 if lights_on else 0.55                 # ánh sáng tự nhiên còn lại khi tắt đèn

        img = base.copy()
        draw_screen(img, projector_on, 1.0, t)       # độ tối toàn cảnh được nhân ở dòng dưới
        img = (img.astype(np.float32) * factor).astype(np.uint8) if not lights_on else img
        cv2.rectangle(img, WINDOW[:2], WINDOW[2:], (235, 225, 210), -1)        # cửa sổ: trời sáng
        cv2.rectangle(img, WINDOW[:2], WINDOW[2:], (80, 80, 80), 3)
        draw_lights(img, lights_on, factor)
        draw_fan(img, angle, speed)

        if t < 15:   # sinh viên (người ngồi yên, chỉ đung đưa nhẹ)
            for k, (cx, bottom) in enumerate(((330, 690), (600, 700), (1010, 700))):
                paste(img, people[k % len(people)], cx, bottom, 330, sway=3 * math.sin(t * 1.3 + k))
        if 75 <= t < 85:   # bảo vệ đi qua
            paste(img, people[-1], 300 + (t - 75) * 25, 700, 360)

        noise = rng.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vw.write(img)
    vw.release()

    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "start_s", "end_s", "devices", "note"])
        w.writerow([a.room, 15.0, 61.0, "light_1;light_2;fan_1", "sinh viên về, quên tắt đèn và quạt"])
    with open(out / "frame_gt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity", "start_s", "end_s"])
        w.writerows([["person", 0, 15], ["person", 75, 85], ["light_1", 0, 50], ["light_2", 0, 50],
                     ["fan_1", 0, 59], ["projector_1", 0, 12]])
    print(f"[demo] đã ghi {n} khung → {out / 'demo_classroom.mp4'}")


if __name__ == "__main__":
    main()
