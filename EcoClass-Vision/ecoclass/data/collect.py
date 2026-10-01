"""Thu thập khung hình để xây dựng bộ dữ liệu (chỉ dùng tại phòng thử nghiệm đã được đồng ý
và có thông báo công khai). Tên file = thời điểm chụp (ms, epoch) để đồng bộ với cảm biến điện năng.

    python -m ecoclass.data.collect --source rtsp://... --room A1-101 --every 10 --out datasets/raw
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ..edge.capture import FrameSource


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--room", required=True)
    ap.add_argument("--every", type=float, default=10.0, help="giây giữa hai ảnh")
    ap.add_argument("--out", default="datasets/raw")
    ap.add_argument("--hours", type=float, default=0, help="dừng sau N giờ (0 = chạy mãi)")
    a = ap.parse_args()

    out = Path(a.out) / a.room
    out.mkdir(parents=True, exist_ok=True)
    src = FrameSource(a.source, fps=1.0 / a.every)
    t_end = time.time() + a.hours * 3600 if a.hours else None
    n = 0
    for fr in src:
        cv2.imwrite(str(out / f"{int(fr.ts * 1000)}.jpg"), fr.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n += 1
        if n % 50 == 0:
            print(f"[collect] {n} ảnh")
        if t_end and time.time() > t_end:
            break


if __name__ == "__main__":
    main()
