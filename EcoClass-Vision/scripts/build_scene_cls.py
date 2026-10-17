"""Tạo dữ liệu huấn luyện bộ phân lớp BẬT/TẮT toàn khung từ video phòng học có nhãn theo video.

Với mỗi khung lấy mẫu (``--fps``), lưu hai ảnh PNG cỡ 384x224 (PNG: không nén mất màu, giữ tín hiệu
chuyển động của ảnh chồng thời gian):
  rgb/<video>_<i>.png      — ảnh màu (bộ phân lớp đèn)
  tstack/<video>_<i>.png   — ảnh chồng thời gian [t-2dt, t-dt, t] (bộ phân lớp quạt, xem ecoclass/edge/tstack.py)
và index.csv (đường dẫn, video, phòng, nhãn light/fan/person, split).

Chia train/val THEO VIDEO (không chia khung của cùng video) để đo khả năng tổng quát hoá thật.
Mặc định giữ lại 4 video làm val, phủ cả BẬT và TẮT của đèn và quạt ở cả hai phòng.

    python scripts/build_scene_cls.py
    python scripts/build_scene_cls.py --val none        # dùng toàn bộ video để huấn luyện bản cuối
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoclass.edge.tstack import SIZE, make_tstack, person_mask_from_result, to_gray_small  # noqa: E402

DEFAULT_VAL = ["r1_172", "r1_211", "r2_224", "r2_239"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="configs/classroom_labels.csv")
    ap.add_argument("--root", default="datasets/classroom_videos")
    ap.add_argument("--out", default="datasets/scene_cls")
    ap.add_argument("--fps", type=float, default=10.0, help="tần số lấy mẫu khung")
    ap.add_argument("--dt", type=float, default=0.1, help="khoảng cách thời gian giữa các khung trong ảnh chồng (s)")
    ap.add_argument("--val", nargs="+", default=DEFAULT_VAL, help="tên video làm val, hoặc 'none'")
    ap.add_argument("--person-model", default="yolo11n-seg.pt")
    a = ap.parse_args()
    val = set() if a.val == ["none"] else set(a.val)
    # Mặt nạ người (YOLO11-seg, như pipeline) → xoá chuyển động của người khỏi ảnh chồng thời gian.
    from ultralytics import YOLO
    seg = YOLO(a.person_model)

    out = Path(a.out)
    for sub in ("rgb", "tstack"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))
    index = []
    for r in rows:
        name = Path(r["video"]).stem
        cap = cv2.VideoCapture(os.path.relpath(Path(a.root) / r["video"]))   # tránh đường dẫn có dấu
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(native / a.fps))
        lag = max(1, round(native * a.dt))
        grays, i, n = [], 0, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            grays.append(to_gray_small(f))
            grays = grays[-(2 * lag + 1):]
            if i % step == 0 and len(grays) == 2 * lag + 1:
                rgb = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
                res = seg.predict(f, classes=[0], conf=0.25, verbose=False)[0]
                ts = make_tstack(grays[0], grays[lag], grays[-1], person_mask_from_result(res, f.shape[:2]))
                fn = f"{name}_{i:04d}.png"
                cv2.imwrite(str(out / "rgb" / fn), rgb)
                cv2.imwrite(str(out / "tstack" / fn), ts)
                index.append({"file": fn, "video": name, "room": r["room"], "light": r["light"], "fan": r["fan"],
                              "person": r["person"], "split": "val" if name in val else "train"})
                n += 1
            i += 1
        cap.release()
        print(f"{name}: {n} mẫu ({'val' if name in val else 'train'})")
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    n_val = sum(x["split"] == "val" for x in index)
    print(f"[scene_cls] {len(index)} mẫu (train={len(index) - n_val}, val={n_val}) → {out / 'index.csv'}")


if __name__ == "__main__":
    main()
