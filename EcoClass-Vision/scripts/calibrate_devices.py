"""Hiệu chuẩn ngưỡng cho đèn/quạt: đo đặc trưng khi thiết bị TẮT và khi BẬT rồi ghi vào cấu hình.

    # 1) Tắt hết thiết bị, phòng vắng:
    python scripts/calibrate_devices.py --config configs/room_A1-101.yaml --state off
    # 2) Bật hết thiết bị:
    python scripts/calibrate_devices.py --config configs/room_A1-101.yaml --state on

Đèn: lưu ref_on_ratio / ref_off_ratio → ngưỡng = trung điểm.
Quạt: lưu sharp_off (độ nét lúc đứng yên) và flow_threshold = trung điểm năng lượng dòng quang học.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecoclass.config import load_room_config, save_room_config  # noqa: E402
from ecoclass.edge.capture import FrameSource  # noqa: E402
from ecoclass.edge.devices import FanDetector, LightDetector, build_detector, make_scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", choices=["on", "off"], required=True)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--source")
    a = ap.parse_args()

    cfg = load_room_config(a.config)
    dets = {d.id: build_detector(d) for d in cfg.devices}
    feats: dict[str, list[dict]] = {k: [] for k in dets}
    src = FrameSource(cfg.resolve(a.source or cfg.source), fps=cfg.fps)
    t_first = None
    for fr in src:
        t_first = t_first or fr.ts
        scene = make_scene(fr.image, [])
        for k, det in dets.items():
            feats[k].append(det.measure(fr.image, scene, fr.ts)[2])
        if fr.ts - t_first >= a.seconds:
            break

    for d in cfg.devices:
        f = [x for x in feats[d.id] if x]
        if isinstance(dets[d.id], LightDetector):
            r = float(np.median([x["ratio"] for x in f]))
            d.params[f"ref_{a.state}_ratio"] = round(r, 3)
            print(f"{d.id}: ratio({a.state}) = {r:.3f}")
        elif isinstance(dets[d.id], FanDetector):
            fl = [x["flow"] for x in f if "flow" in x]
            sh = [x["sharp"] for x in f if "sharp" in x]
            if a.state == "off":
                d.params["sharp_off"] = round(float(np.median(sh)), 1)
                d.params["flow_off"] = round(float(np.percentile(fl, 95)), 3)
            else:
                d.params["flow_on"] = round(float(np.median(fl)), 3)
            if "flow_off" in d.params and "flow_on" in d.params:
                d.params["flow_threshold"] = round(0.5 * (d.params["flow_off"] + d.params["flow_on"]), 3)
            print(f"{d.id}: flow median={np.median(fl):.3f} sharp median={np.median(sh):.1f}")
    save_room_config(cfg, a.config)
    print("[calibrate] đã cập nhật", a.config)


if __name__ == "__main__":
    main()
