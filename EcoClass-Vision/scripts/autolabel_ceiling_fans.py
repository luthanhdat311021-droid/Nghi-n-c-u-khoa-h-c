"""Tạo bộ dữ liệu YOLO chỉ gồm loại QUẠT TRẦN có trong các video phòng học (datasets/classroom_videos),
nhãn box sinh tự động bằng YOLO-World ("ceiling fan") rồi lọc:
  * ổn định theo thời gian: box phải lặp lại (IoU >= 0.3) ở đa số khung lân cận (±0,5 s) → bỏ phát hiện lẻ tẻ;
  * box nhỏ nằm gần trọn trong box lớn hơn (chỉ ôm động cơ khi cánh đang quay nhòe) → bỏ, giữ box ôm cả cánh;
  * box > 35% khung (giống DeviceLocator) → bỏ.
Ảnh camera không có quạt trần (vd datasets/scene_cls_imou/rgb) được thêm làm ảnh nền (nhãn rỗng) để mô hình
học "không có loại quạt này thì không có box". Lớp duy nhất tên "fan" để FanAutoDetector dùng thẳng.
Chỉ ghi vào thư mục --out MỚI (dừng nếu đã tồn tại) — không sửa dữ liệu cũ.

    python scripts/autolabel_ceiling_fans.py --out datasets/ceiling_fan
    python scripts/train_devices.py --data datasets/ceiling_fan/data.yaml --model yolo11n.pt --name ceiling_fan \
        --out models/ceiling_fan/ceiling_fan_yolo11n.pt
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VAL = ["r1_172", "r1_211", "r2_224", "r2_239"]   # như build_scene_cls.py: chia theo video


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa một box a và mảng box b (xyxy)."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def drop_nested(boxes: np.ndarray, thr: float) -> np.ndarray:
    """Bỏ box nằm >= thr diện tích bên trong một box lớn hơn."""
    if len(boxes) < 2:
        return boxes
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    for i, a in enumerate(boxes):
        x1, y1 = np.maximum(a[0], boxes[:, 0]), np.maximum(a[1], boxes[:, 1])
        x2, y2 = np.minimum(a[2], boxes[:, 2]), np.minimum(a[3], boxes[:, 3])
        inside = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None) / max(area[i], 1e-9)
        bigger = area > area[i]
        if not np.any(bigger & (inside >= thr)):
            keep.append(i)
    return boxes[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="datasets/classroom_videos")
    ap.add_argument("--out", default="datasets/ceiling_fan")
    ap.add_argument("--model", default="weights/yolov8s-worldv2.pt")
    ap.add_argument("--prompt", default="ceiling fan")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--every", type=int, default=5, help="lưu 1/N khung")
    ap.add_argument("--window", type=float, default=0.5, help="nửa cửa sổ kiểm tra ổn định (s)")
    ap.add_argument("--persist", type=float, default=0.6, help="tỉ lệ khung lân cận phải có box trùng")
    ap.add_argument("--nested", type=float, default=0.6)
    ap.add_argument("--max-area", type=float, default=0.35)
    ap.add_argument("--val", nargs="+", default=DEFAULT_VAL)
    ap.add_argument("--negatives", nargs="*", default=["datasets/scene_cls_imou/rgb"],
                    help="thư mục ảnh KHÔNG có quạt trần → ảnh nền (nhãn rỗng)")
    ap.add_argument("--neg-count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} đã tồn tại — chọn --out khác (không ghi đè dữ liệu cũ).")
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    (out / "preview").mkdir()

    from ultralytics import YOLOWorld
    world = YOLOWorld(a.model)
    world.set_classes([a.prompt])

    videos = sorted(Path(a.videos).rglob("*.mp4"))
    stats = {"train": [0, 0], "val": [0, 0]}
    for vp in videos:
        name, split = vp.stem, ("val" if vp.stem in a.val else "train")
        cap = cv2.VideoCapture(os.path.relpath(vp))          # tránh đường dẫn có dấu
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames, dets = [], []
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                r = world.predict(f, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                frames.append(f)
                dets.append(r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4)))
            i += 1
        cap.release()
        H, W = frames[0].shape[:2]
        k = max(1, round(a.window * fps / a.every))
        n_box = 0
        for j, (f, d) in enumerate(zip(frames, dets)):
            nb = [dets[t] for t in range(max(0, j - k), min(len(dets), j + k + 1)) if t != j]
            keep = []
            for b in d:
                hits = sum(1 for o in nb if len(o) and iou(b, o).max() >= 0.3)
                area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
                if nb and hits / len(nb) >= a.persist and area <= a.max_area:
                    keep.append(b)
            boxes = drop_nested(np.array(keep).reshape(-1, 4), a.nested)
            fn = f"{name}_{j * a.every:04d}"
            cv2.imwrite(str(out / "images" / split / f"{fn}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out / "labels" / split / f"{fn}.txt", "w") as fh:
                for x1, y1, x2, y2 in boxes:
                    fh.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            n_box += len(boxes)
            if j == len(frames) // 2:          # ảnh xem trước để rà soát nhãn
                pv = f.copy()
                for x1, y1, x2, y2 in boxes.astype(int):
                    cv2.rectangle(pv, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imwrite(str(out / "preview" / f"{fn}.jpg"), pv)
        stats[split][0] += len(frames)
        stats[split][1] += n_box
        print(f"{name} ({split}): {len(frames)} ảnh, {n_box} box ({n_box / max(1, len(frames)):.1f}/ảnh)", flush=True)

    # Ảnh nền không có quạt trần: các phiên *_val sang val, còn lại sang train.
    rng = random.Random(a.seed)
    for d in a.negatives:
        files = sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg"))
        val = [p for p in files if "_val_" in p.name]
        tr = [p for p in files if "_val_" not in p.name]
        picks = {"train": rng.sample(tr, min(a.neg_count, len(tr))),
                 "val": rng.sample(val, min(a.neg_count // 2, len(val)))}
        for split, ps in picks.items():
            for p in ps:
                dst = out / "images" / split / f"neg_{Path(d).parent.name}_{p.name}"
                shutil.copy(p, dst)
                (out / "labels" / split / (dst.stem + ".txt")).touch()
            stats[split][0] += len(ps)
        print(f"ảnh nền từ {d}: train {len(picks['train'])}, val {len(picks['val'])}", flush=True)

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: fan\n", encoding="utf-8")
    for s, (n, b) in stats.items():
        print(f"[ceiling_fan] {s}: {n} ảnh, {b} box")
    print(f"[ceiling_fan] → {out / 'data.yaml'}  (xem nhãn mẫu: {out / 'preview'})")


if __name__ == "__main__":
    main()
