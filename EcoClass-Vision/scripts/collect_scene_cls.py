"""Thu ảnh huấn luyện bộ phân lớp đèn BẬT/TẮT toàn khung trực tiếp từ camera (RTSP / webcam), mỗi lần
một phiên có nhãn cố định. Cùng định dạng với build_scene_cls.py (rgb/<phiên>_<i>.png cỡ 384x224 + index.csv)
để train_scene_cls.py dùng được, nhưng ghi vào thư mục RIÊNG và chỉ THÊM dòng mới — không sửa dữ liệu cũ.

    python scripts/collect_scene_cls.py --source rtsp://... --session imou_on1 --light 1 --seconds 60
    python scripts/collect_scene_cls.py --source rtsp://... --session imou_off1 --light 0 --seconds 60 --split val
Trong lúc thu: giữ nguyên trạng thái đèn, đi lại/ngồi/đứng trong khung để ảnh đa dạng.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoclass.edge.capture import FrameSource  # noqa: E402
from ecoclass.edge.tstack import SIZE  # noqa: E402

FIELDS = ["file", "video", "room", "light", "fan", "person", "split"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--session", required=True, help="tên phiên (duy nhất), vd imou_on1")
    ap.add_argument("--light", required=True, choices=["0", "1"], help="nhãn đèn trong suốt phiên: 1 = bật")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--fps", type=float, default=4.0, help="tần số lưu ảnh (thấp → ảnh ít trùng lặp)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--room", default="imou")
    ap.add_argument("--max-side", type=int, default=640, help="thu nhỏ như pipeline (max_side)")
    ap.add_argument("--out", default="datasets/scene_cls_imou")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    index = out / "index.csv"
    rows = list(csv.DictReader(open(index, encoding="utf-8"))) if index.exists() else []
    if any(r["video"] == a.session for r in rows):
        sys.exit(f"Phiên '{a.session}' đã có trong {index} — chọn tên khác (không ghi đè dữ liệu cũ).")

    new = []
    t_end = time.time() + a.seconds
    for fr in FrameSource(a.source, fps=a.fps, max_side=a.max_side):
        if time.time() >= t_end:
            break
        # Pipeline bỏ qua bộ phân lớp khi cảnh quá tối (mean V < 25) → không cần mẫu đó.
        if cv2.cvtColor(fr.image, cv2.COLOR_BGR2HSV)[..., 2].mean() < 25:
            continue
        fn = f"{a.session}_{fr.index:04d}.png"
        cv2.imwrite(str(out / "rgb" / fn), cv2.resize(fr.image, SIZE, interpolation=cv2.INTER_AREA))
        new.append({"file": fn, "video": a.session, "room": a.room, "light": a.light, "fan": "", "person": "",
                    "split": a.split})
        if len(new) % 20 == 0:
            print(f"[collect] {len(new)} ảnh", flush=True)
    with open(index, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not rows:
            w.writeheader()
        w.writerows(new)
    print(f"[collect] {a.session}: {len(new)} ảnh light={a.light} ({a.split}) → {index}", flush=True)


if __name__ == "__main__":
    main()
