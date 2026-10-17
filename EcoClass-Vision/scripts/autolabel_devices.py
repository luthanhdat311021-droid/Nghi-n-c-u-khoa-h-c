"""Tự gán nhãn (nhãn giả) đèn/quạt cho phiên ghi webcam → bộ dữ liệu YOLO để huấn luyện bộ phát hiện.

Chạy tuần tự qua các khung (theo thời gian) với:
  - YOLO11n-seg: mặt nạ người (loại người khỏi vùng đo);
  - lights_auto: các máng đèn đang sáng (đã lọc cửa sổ/tường cháy sáng) → lớp 0 "light";
  - fan_auto: vùng cánh quạt đang quay (cần camera đứng yên vài giây) → lớp 1 "fan".
Chỉ giữ khung có kết quả đo tin cậy cho cả hai (không lia camera, không đang khởi động), để tránh
nhãn thiếu. Chia train/val theo khối thời gian liên tiếp (tránh rò rỉ giữa khung gần giống nhau).
Lưu ảnh xem trước các nhãn để rà soát.

    python scripts/autolabel_devices.py --frames datasets/raw/session1 --out datasets/devices --fps 10
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoclass.config import DeviceConfig, PersonConfig  # noqa: E402
from ecoclass.edge.devices import build_detector, make_scene  # noqa: E402
from ecoclass.edge.person import PersonCounter  # noqa: E402

CLASSES = ["fan", "window", "light"]       # trùng thứ tự lớp với datasets/oi_devices (0 fan, 1 window) + light


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs="+", required=True, help="một hoặc nhiều thư mục phiên ghi")
    ap.add_argument("--out", default="datasets/devices")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--block", type=float, default=10.0, help="độ dài khối thời gian khi chia train/val (s)")
    ap.add_argument("--keep-every", type=int, default=2, help="giữ 1/N khung (giảm khung gần trùng)")
    ap.add_argument("--model", help="YOLO giai đoạn 1 (fan, window) để đề xuất box quạt/cửa sổ")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--merge", help="data.yaml của bộ dữ liệu công khai để gộp chung khi huấn luyện")
    a = ap.parse_args()
    locator = None
    if a.model:
        from ecoclass.edge.device_yolo import DeviceLocator
        locator = DeviceLocator(a.model, conf=a.conf)

    k = a.fps / 5.0   # tham số đếm khung theo 5 fps → quy đổi
    light_cfg = DeviceConfig(id="lights", type="lights_auto", roi=[0, 0, 1, 1], on_frames=1, off_frames=1,
                             params={"v_min": 240, "min_area": 0.002, "max_area": 0.12, "fill_min": 0.4,
                                     "s_mean_max": 12, "max_neighbors": 0})
    fan_cfg = DeviceConfig(id="fans", type="fan_auto", roi=[0, 0, 1, 1], on_frames=1, off_frames=1,
                           params={"window": int(8 * k), "min_frames": int(5 * k), "tstd_pix": 4.0,
                                   "heat_alpha": 0.3 / k, "heat_on": 0.5, "min_area": 0.004, "max_aspect": 3.0})
    pc = PersonCounter(PersonConfig(model="yolo11n-seg.pt", conf_occupied=0.3, conf_empty=0.3))

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)
    prev_dir = out / "preview"
    prev_dir.mkdir()

    rnd = random.Random(0)
    stats = {"frames": 0, "kept": 0, "light": 0, "fan": 0, "skipped_unreliable": 0}
    previews = []
    for d in a.frames:
        files = sorted(Path(d).glob("*.jpg"), key=lambda p: int(p.stem))
        if not files:
            continue
        lights, fans = build_detector(light_cfg), build_detector(fan_cfg)
        t_first = int(files[0].stem) / 1000
        block_split: dict[int, str] = {}
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            ts = int(f.stem) / 1000
            stats["frames"] += 1
            pr = pc.update(img, ts, occupied=True)
            dev = locator(img) if locator else None
            # Đèn luôn dò theo luật (loại vùng nằm trong cửa sổ nếu có YOLO); quạt/cửa sổ lấy từ YOLO nếu có.
            sc = make_scene(img, pr.boxes, pr.mask, device_boxes={"window": dev["window"]} if dev else None)
            rl = lights.update(img, sc, ts)
            if dev is None:
                rf = fans.update(img, sc, ts)
                if rf.raw == "UNKNOWN" or rl.raw == "UNKNOWN":
                    stats["skipped_unreliable"] += 1
                    continue
                fan_boxes, win_boxes = fans.blobs, []
            else:
                fan_boxes, win_boxes = dev["fan"], dev["window"]
            if i % a.keep_every:
                continue
            blk = int((ts - t_first) // a.block)
            if blk not in block_split:
                block_split[blk] = "val" if rnd.random() < a.val else "train"
            sp = block_split[blk]
            name = f"{Path(d).name}_{f.stem}"
            lines = []
            for cls, boxes in ((0, fan_boxes), (1, win_boxes), (2, lights.blobs)):
                for x1, y1, x2, y2 in boxes:
                    lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
                    stats[CLASSES[cls]] += 1
            shutil.copy(f, out / "images" / sp / f"{name}.jpg")
            (out / "labels" / sp / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats["kept"] += 1
            if len(previews) < 400 and stats["kept"] % 8 == 0:
                v = img.copy()
                h, w = v.shape[:2]
                for cls, boxes, col in ((0, fan_boxes, (0, 0, 255)), (1, win_boxes, (255, 128, 0)),
                                        (2, lights.blobs, (0, 255, 255))):
                    for x1, y1, x2, y2 in boxes:
                        cv2.rectangle(v, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, 2)
                        cv2.putText(v, CLASSES[cls], (int(x1 * w), int(y1 * h) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                previews.append(cv2.resize(v, (320, 240)))

    for i in range(0, len(previews), 12):
        tiles = previews[i:i + 12] + [np.zeros((240, 320, 3), np.uint8)] * (12 - len(previews[i:i + 12]))
        grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)])
        cv2.imwrite(str(prev_dir / f"preview_{i // 12:02d}.jpg"), grid)

    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    if a.merge:
        # Gộp với bộ dữ liệu công khai: ảnh công khai không có nhãn "light" (thiếu nhãn, chấp nhận được
        # vì đèn huỳnh quang trần hiếm gặp trong các ảnh đó).
        import yaml
        base = yaml.safe_load(Path(a.merge).read_text(encoding="utf-8"))
        root = Path(base["path"])
        train = [str((root / base["train"]).as_posix()), str((out / "images/train").resolve().as_posix())]
        val = [str((root / base["val"]).as_posix()), str((out / "images/val").resolve().as_posix())]
        (out / "data.yaml").write_text(f"train: {train}\nval: {val}\nnames:\n{names}", encoding="utf-8")
    else:
        (out / "data.yaml").write_text(
            f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}", encoding="utf-8")
    n_tr = len(list((out / "images" / "train").glob("*.jpg")))
    n_va = len(list((out / "images" / "val").glob("*.jpg")))
    print(f"[autolabel] {stats} | train={n_tr} val={n_va} → {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
