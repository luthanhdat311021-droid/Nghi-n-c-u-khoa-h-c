"""Chạy quy trình xử lý tại biên cho một phòng học.

    python scripts/run_edge.py --config configs/room_demo.yaml --show
    python scripts/run_edge.py --config configs/room_demo.yaml --publisher stdout --save-video runs/demo_debug.mp4
    # Ablation:
    python scripts/run_edge.py --config configs/room_demo.yaml --no-temporal --events-out runs/ablation_no_fsm.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecoclass.config import load_room_config  # noqa: E402
from ecoclass.edge.pipeline import run  # noqa: E402
from ecoclass.edge.publisher import Publisher  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", help="ghi đè nguồn video trong cấu hình")
    ap.add_argument("--publisher", choices=["http", "mqtt", "stdout", "none"], help="ghi đè publisher.mode")
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ gỡ lỗi (người đã làm mờ)")
    ap.add_argument("--save-video", help="lưu video gỡ lỗi (người đã làm mờ)")
    ap.add_argument("--frame-log", help="ghi log từng khung (JSONL) để phân tích")
    ap.add_argument("--events-out", help="ghi sự kiện (JSONL) để đánh giá mức sự kiện")
    ap.add_argument("--start", help="thời điểm bắt đầu của file video, ISO (vd 2026-10-01T16:30:00) — dùng cho thời khoá biểu")
    ap.add_argument("--realtime", action="store_true", help="phát file video theo thời gian thực")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--select-roi", nargs="+", metavar="DEVICE_ID",
                    help="kéo chuột chọn ROI cho các thiết bị này trên khung đầu tiên")
    ap.add_argument("--no-track", action="store_true", help="ablation: bỏ ByteTrack")
    ap.add_argument("--no-temporal", action="store_true", help="ablation: bỏ bộ suy luận theo thời gian")
    a = ap.parse_args()

    cfg = load_room_config(a.config)
    if a.source:
        cfg.source = a.source
    if a.publisher:
        cfg.publisher.mode = a.publisher
    pub = None if cfg.publisher.mode == "none" else Publisher(cfg.publisher, cfg.room_id)
    t0 = datetime.fromisoformat(a.start).timestamp() if a.start else None

    events = run(cfg, pub, show=a.show, save_video=a.save_video, use_tracker=not a.no_track,
                 temporal=not a.no_temporal, frame_log=a.frame_log, t0=t0, realtime=a.realtime,
                 max_frames=a.max_frames, select_roi=a.select_roi)
    if a.events_out:
        Path(a.events_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.events_out, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[edge] {len(events)} sự kiện → {a.events_out}")


if __name__ == "__main__":
    main()
