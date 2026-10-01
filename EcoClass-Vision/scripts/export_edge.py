"""Tối ưu mô hình phát hiện người cho Raspberry Pi 5 (ONNX / NCNN, lượng tử hoá INT8) và đo tốc độ.

    python scripts/export_edge.py --weights yolo11n.pt --formats ncnn onnx --imgsz 416
    python scripts/export_edge.py --benchmark yolo11n_ncnn_model --imgsz 416      # chạy trên Pi để đo FPS

Trên Raspberry Pi 5, NCNN thường nhanh nhất cho YOLO11n; giảm imgsz 640 → 416/320 tăng FPS
đáng kể với camera góc rộng gần (người chiếm diện tích lớn trong khung).
Lượng tử hoá INT8 (--int8) cần tập hiệu chuẩn --data (định dạng YOLO).
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    import numpy as np
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--formats", nargs="+", default=["ncnn", "onnx"])
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data", help="data.yaml để hiệu chuẩn INT8")
    ap.add_argument("--benchmark", help="đường dẫn mô hình đã xuất để đo tốc độ")
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()

    if a.benchmark:
        m = YOLO(a.benchmark, task="detect")
        img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        for _ in range(5):
            m.predict(img, imgsz=a.imgsz, classes=[0], verbose=False)
        t = time.perf_counter()
        for _ in range(a.runs):
            m.predict(img, imgsz=a.imgsz, classes=[0], verbose=False)
        dt = (time.perf_counter() - t) / a.runs
        print(f"[bench] {a.benchmark} imgsz={a.imgsz}: {dt * 1000:.1f} ms/khung → {1 / dt:.1f} FPS")
        return

    model = YOLO(a.weights)
    for fmt in a.formats:
        kw = dict(format=fmt, imgsz=a.imgsz)
        if a.int8:
            kw.update(int8=True, data=a.data)
        if fmt == "onnx":
            kw.update(simplify=True, opset=17)
        path = model.export(**kw)
        print(f"[export] {fmt} → {path}")


if __name__ == "__main__":
    main()
