"""Đánh giá mức khung trên bộ video phòng học có nhãn (đèn/quạt/người theo từng video).

Mỗi video chạy một EdgePipeline mới. Bỏ qua ``--warmup`` giây đầu (bộ đếm trễ, cửa sổ quạt).
Một khung được tính đúng khi trạng thái đã làm trễ khớp nhãn; UNKNOWN tính là sai.

    python scripts/eval_videos.py --config configs/room_classroom.yaml
    python scripts/eval_videos.py --config configs/room_classroom.yaml --out runs/eval_classroom.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecoclass.config import load_room_config  # noqa: E402
from ecoclass.edge.capture import FrameSource  # noqa: E402
from ecoclass.edge.pipeline import EdgePipeline  # noqa: E402

TASKS = {"light": "lights", "fan": "fans"}     # cột nhãn → id thiết bị trong cấu hình


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", default="configs/classroom_labels.csv")
    ap.add_argument("--root", default="datasets/classroom_videos")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--only", nargs="+", help="chỉ chạy các video có tên chứa chuỗi này")
    ap.add_argument("--out", help="ghi bảng kết quả theo video (CSV)")
    a = ap.parse_args()

    base = load_room_config(a.config)
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))
    if a.only:
        rows = [r for r in rows if any(s in r["video"] for s in a.only)]
    results = []
    for r in rows:
        cfg = copy.deepcopy(base)
        src = os.path.relpath(Path(a.root) / r["video"])     # OpenCV/Windows: tránh đường dẫn có dấu
        pipe = EdgePipeline(cfg, None)
        hits = {k: 0 for k in (*TASKS, "person")}
        n = 0
        t_first = None
        for fr in FrameSource(src, fps=cfg.fps, t0=0.0):
            out = pipe.process(fr.image, fr.ts)
            t_first = fr.ts if t_first is None else t_first
            if fr.ts - t_first < a.warmup:
                continue
            n += 1
            for col, dev in TASKS.items():
                hits[col] += out["states"].get(dev) == ("ON" if r[col] == "1" else "OFF")
            hits["person"] += (out["person"].n > 0) == (r["person"] == "1")
        pipe.close()
        acc = {k: v / max(n, 1) for k, v in hits.items()}
        results.append({"video": r["video"], "frames": n, **{f"acc_{k}": round(v, 3) for k, v in acc.items()}})
        print(f"{r['video']:22s} n={n:3d}  " + "  ".join(f"{k}={v:.2f}" for k, v in acc.items())
              + f"   (gt L={r['light']} F={r['fan']} P={r['person']})", flush=True)

    tot = sum(x["frames"] for x in results)
    print("-" * 70)
    for k in (*TASKS, "person"):
        frame_acc = sum(x[f"acc_{k}"] * x["frames"] for x in results) / max(tot, 1)
        vid_ok = sum(x[f"acc_{k}"] >= 0.5 for x in results)
        print(f"{k:7s} độ chính xác mức khung = {frame_acc:.3f}   video đúng (đa số khung) = {vid_ok}/{len(results)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    main()
