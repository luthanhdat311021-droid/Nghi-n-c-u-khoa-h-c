"""Tự gán nhãn bằng cảm biến điện năng (giám sát yếu – weak supervision).

1. Nhãn trạng thái thiết bị: mỗi ảnh (tên file = epoch ms) được ghép với số đo công suất
   gần nhất của cảm biến gắn với thiết bị (cùng device_id trong cấu hình phòng).
   P >= on_power_w → on, P <= off_power_w → off, ở giữa hoặc gần thời điểm chuyển trạng thái → bỏ qua.
   ROI của thiết bị được cắt vào datasets/roi_cls/<type>/<split>/<on|off>/ để huấn luyện MobileNetV3.
2. Nhãn người (tuỳ chọn --person): gán nhãn giả bằng mô hình lớn (YOLO11x) theo định dạng YOLO
   để người gán nhãn chỉ cần rà soát/sửa trong CVAT/Label Studio, thay vì vẽ từ đầu.

    python -m ecoclass.data.autolabel --frames datasets/raw/A1-101 --sensor datasets/sensor_A1-101.csv \
        --config configs/room_A1-101.yaml --out datasets --person
"""
from __future__ import annotations

import argparse
import bisect
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2

from ..config import load_room_config
from ..edge.geometry import crop


def load_sensor(path: str) -> dict[str, tuple[list[float], list[float]]]:
    """device_id -> (ts đã sắp xếp, power)."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows[r["device_id"]].append((float(r["ts"]), float(r["power"])))
            except (TypeError, ValueError):
                continue
    out = {}
    for dev, lst in rows.items():
        lst.sort()
        out[dev] = ([t for t, _ in lst], [p for _, p in lst])
    return out


def label_at(ts_list, p_list, ts, on_w, off_w, tol, guard) -> str | None:
    i = bisect.bisect_left(ts_list, ts)
    cands = [j for j in (i - 1, i) if 0 <= j < len(ts_list)]
    if not cands:
        return None
    j = min(cands, key=lambda k: abs(ts_list[k] - ts))
    if abs(ts_list[j] - ts) > tol:
        return None

    def st(p):
        return "on" if p >= on_w else "off" if p <= off_w else None

    s = st(p_list[j])
    if s is None:
        return None
    # Loại các ảnh gần thời điểm chuyển trạng thái (quạt còn quay chậm dần, đèn đang mồi...)
    lo = bisect.bisect_left(ts_list, ts - guard)
    hi = bisect.bisect_right(ts_list, ts + guard)
    if any(st(p) != s for p in p_list[lo:hi]):
        return None
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="thư mục ảnh, tên file = epoch ms")
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--tol", type=float, default=3.0, help="lệch thời gian tối đa ảnh–cảm biến (s)")
    ap.add_argument("--guard", type=float, default=10.0, help="bỏ ảnh trong ±guard giây quanh lúc chuyển trạng thái")
    ap.add_argument("--on-ratio", type=float, default=0.3, help="P >= on_ratio * power_w → on")
    ap.add_argument("--off-w", type=float, default=2.0, help="P <= off_w → off")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--person", action="store_true", help="tạo nhãn giả cho người bằng YOLO11x")
    ap.add_argument("--person-model", default="yolo11x.pt")
    a = ap.parse_args()

    cfg = load_room_config(a.config)
    sensor = load_sensor(a.sensor)
    frames = sorted(Path(a.frames).glob("*.jpg"), key=lambda p: int(p.stem))
    rnd = random.Random(0)
    # Chia train/val theo ngày để tránh rò rỉ giữa các ảnh liên tiếp gần giống nhau.
    days = sorted({int(p.stem) // 86_400_000 for p in frames})
    val_days = set(rnd.sample(days, max(1, int(len(days) * a.val)))) if len(days) > 1 else set()

    out = Path(a.out)
    label_rows, counts = [], defaultdict(int)
    for p in frames:
        ts = int(p.stem) / 1000.0
        split = "val" if (int(p.stem) // 86_400_000) in val_days or (not val_days and rnd.random() < a.val) else "train"
        img = None
        row = {"file": p.name, "ts": ts, "split": split}
        for d in cfg.devices:
            if d.id not in sensor:
                continue
            on_w = max(a.off_w + 1, a.on_ratio * d.power_w) if d.power_w else 5.0
            s = label_at(*sensor[d.id], ts, on_w, a.off_w, a.tol, a.guard)
            row[d.id] = s or ""
            if s is None:
                continue
            img = img if img is not None else cv2.imread(str(p))
            dst = out / "roi_cls" / d.type / split / s
            dst.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst / f"{cfg.room_id}_{d.id}_{p.stem}.jpg"), crop(img, d.roi))
            counts[(d.type, s)] += 1
        label_rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    keys = ["file", "ts", "split"] + [d.id for d in cfg.devices]
    with open(out / f"device_labels_{cfg.room_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(label_rows)
    print("[autolabel] nhãn thiết bị:", dict(counts))

    if a.person:
        from ultralytics import YOLO

        model = YOLO(a.person_model)
        n_box = 0
        for row in label_rows:
            p = Path(a.frames) / row["file"]
            r = model.predict(str(p), classes=[0], conf=0.35, verbose=False)[0]
            img_dir = out / "person" / "images" / row["split"]
            lbl_dir = out / "person" / "labels" / row["split"]
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            name = f"{cfg.room_id}_{p.stem}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), r.orig_img)
            with open(lbl_dir / f"{name}.txt", "w") as f:
                for b in r.boxes.xywhn.cpu().numpy():
                    f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                    n_box += 1
        (out / "person" / "data.yaml").write_text(
            f"path: {(out / 'person').resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
            encoding="utf-8")
        print(f"[autolabel] nhãn giả người: {len(label_rows)} ảnh, {n_box} box → cần rà soát thủ công trước khi huấn luyện")


if __name__ == "__main__":
    main()
