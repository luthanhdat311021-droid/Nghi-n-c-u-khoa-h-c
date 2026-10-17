"""Huấn luyện YOLO11 phát hiện thiết bị (fan, window[, light]) và chép trọng số tốt nhất vào models/.

Giai đoạn 1 — dữ liệu công khai Open Images:
    python scripts/train_devices.py --data datasets/oi_devices/data.yaml --epochs 60
Giai đoạn 2 — tinh chỉnh bằng dữ liệu quay tại phòng học (khởi tạo từ trọng số giai đoạn 1):
    python scripts/train_devices.py --data datasets/classroom/data.yaml --model models/devices_yolo11n.pt --epochs 30 --name classroom_ft

RTX 3050 4 GB: yolo11n, imgsz 640, batch 16 (AMP) chạy được.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="devices_oi")
    ap.add_argument("--out", default="models/devices_yolo11n.pt")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=4,
        project="runs/devices", name=a.name, exist_ok=True, patience=15, cos_lr=True,
        # Webcam phòng học: ánh sáng thay đổi mạnh, camera nghiêng/lật, ảnh nhòe khi lia.
        hsv_v=0.5, hsv_s=0.5, degrees=10, fliplr=0.5, flipud=0.1, mosaic=1.0, close_mosaic=10, erasing=0.3,
        plots=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=a.data, imgsz=a.imgsz, device=a.device, split="val", plots=False)
    names = m.names
    print(f"mAP@0.5={m.box.map50:.4f}  mAP@0.5:0.95={m.box.map:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}")
    for i, c in enumerate(m.box.ap_class_index):
        print(f"  {names[int(c)]:8s} AP@0.5={m.box.ap50[i]:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, a.out)
    print(f"[train] → {a.out}")


if __name__ == "__main__":
    main()
