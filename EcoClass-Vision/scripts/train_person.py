"""Tinh chỉnh YOLO11 phát hiện người trên dữ liệu phòng học (khởi tạo từ trọng số COCO).

    python scripts/train_person.py --data datasets/person/data.yaml --model yolo11n.pt --epochs 100
    python scripts/train_person.py --data datasets/person/data.yaml --eval-only --weights runs/person/yolo11n_classroom/weights/best.pt

RTX 3050 4GB: dùng batch 16 với imgsz 640 cho yolo11n/s.
Tăng cường dữ liệu hướng vào thách thức ánh sáng (HSV, độ sáng) và che khuất (mosaic, erasing).
"""
from __future__ import annotations

import argparse


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="yolo11n_classroom")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights")
    a = ap.parse_args()

    if not a.eval_only:
        model = YOLO(a.model)
        model.train(
            data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project="runs/person", name=a.name, exist_ok=True, patience=20,
            single_cls=True, cos_lr=True,
            hsv_v=0.5, hsv_s=0.6, degrees=0.0, fliplr=0.5, mosaic=1.0, close_mosaic=10, erasing=0.4,
        )
        a.weights = f"runs/person/{a.name}/weights/best.pt"

    m = YOLO(a.weights or a.model).val(data=a.data, imgsz=a.imgsz, device=a.device, split="val")
    print(f"mAP@0.5={m.box.map50:.4f}  mAP@0.5:0.95={m.box.map:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}")


if __name__ == "__main__":
    main()
