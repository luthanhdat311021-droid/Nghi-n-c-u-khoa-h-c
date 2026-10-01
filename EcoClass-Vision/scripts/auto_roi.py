"""Auto-ROI khi lắp đặt: dùng YOLO-World định vị thiết bị bằng câu lệnh văn bản, sinh file cấu hình phòng.

    python scripts/auto_roi.py --source rtsp://... --room A1-101 --out configs/room_A1-101.yaml
    python scripts/auto_roi.py --source data/demo/demo_classroom.mp4 --room A1-101 --out runs/auto_roi.yaml --preview runs/auto_roi.jpg

Nên lấy khung lúc thiết bị đang BẬT và phòng vắng. Kết quả cần kỹ thuật viên rà soát
(ảnh --preview) trước khi đưa vào vận hành; có thể sửa trực tiếp file YAML.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecoclass.config import DeviceConfig, RoomConfig, save_room_config  # noqa: E402
from ecoclass.edge.auto_roi import DEFAULT_POWER_W, DEFAULT_PROMPTS, detect_rois, suggest_ref_roi  # noqa: E402
from ecoclass.edge.capture import FrameSource  # noqa: E402
from ecoclass.edge.geometry import roi_to_px  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--room", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5, help="số khung dùng để gộp kết quả")
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--prompt", action="append", help='thêm câu lệnh dạng "ceiling lamp=light"')
    ap.add_argument("--preview", help="lưu ảnh xem trước các ROI")
    a = ap.parse_args()

    frames = []
    for fr in FrameSource(a.source, fps=0.5):
        frames.append(fr.image)
        if len(frames) >= a.frames:
            break
    if not frames:
        sys.exit("Không đọc được khung hình nào")

    prompts = dict(DEFAULT_PROMPTS)
    for p in a.prompt or []:
        k, v = p.split("=")
        prompts[k.strip()] = v.strip()

    cands = detect_rois(frames, a.model, prompts, conf=a.conf)
    counter: dict[str, int] = {}
    devices = []
    for c in cands:
        counter[c.type] = counter.get(c.type, 0) + 1
        devices.append(DeviceConfig(
            id=f"{c.type}_{counter[c.type]}", type=c.type, roi=c.roi, power_w=DEFAULT_POWER_W.get(c.type, 50),
            ref_roi=suggest_ref_roi(c.roi) if c.type in ("light", "screen", "projector", "monitor", "tv") else None,
            params={"autoroi_prompt": c.prompt, "autoroi_conf": c.conf},
        ))
        print(f"  {devices[-1].id:12s} '{c.prompt}' conf={c.conf:.2f} roi={c.roi}")

    cfg = RoomConfig(room_id=a.room, source=a.source, devices=devices)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_room_config(cfg, a.out)
    print(f"[auto-roi] {len(devices)} thiết bị → {a.out}")

    if a.preview:
        img = frames[-1].copy()
        h, w = img.shape[:2]
        for d in devices:
            x1, y1, x2, y2 = roi_to_px(d.roi, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, d.id, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if d.ref_roi:
                x1, y1, x2, y2 = roi_to_px(d.ref_roi, w, h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.imwrite(a.preview, img)
        print(f"[auto-roi] ảnh xem trước → {a.preview}")


if __name__ == "__main__":
    main()
