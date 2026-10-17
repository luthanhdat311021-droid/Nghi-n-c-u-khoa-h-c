"""Ghi một phiên khung hình từ webcam để làm dữ liệu huấn luyện bộ phát hiện đèn/quạt.

    python scripts/record_session.py --seconds 120 --fps 10 --out datasets/raw/session1

Trong lúc ghi: lia camera CHẬM khắp phòng, dừng 2–3 giây ở mỗi hướng (để bộ gán nhãn tự động
nhận ra quạt đang quay), quét qua đèn, quạt, cửa sổ, tường, người. Nhấn q để dừng sớm.
Tên file = epoch ms. Dữ liệu chỉ lưu trên máy, dùng để huấn luyện.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoclass.edge.capture import FrameSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--flip", default="horizontal")
    ap.add_argument("--out", default="datasets/raw/session1")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = FrameSource(a.source, fps=a.fps, width=640, height=480, flip=a.flip)
    title = "Dang ghi du lieu - lia cham, dung 2-3s moi huong (q = dung)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    t0, n = time.time(), 0
    for fr in src:
        cv2.imwrite(str(out / f"{int(fr.ts * 1000)}.jpg"), fr.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
        left = a.seconds - (time.time() - t0)
        view = cv2.resize(fr.image, None, fx=1.5, fy=1.5)
        cv2.putText(view, f"REC {n} anh  con {left:.0f}s", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(title, view)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")) or left <= 0:
            break
    cv2.destroyAllWindows()
    print(f"[record] {n} ảnh → {out}")


if __name__ == "__main__":
    main()
