"""Đánh giá hệ thống ở hai mức (mục 10 – Cách tiếp cận):

1) Mức sự kiện (tiêu chí chính): so cảnh báo "bỏ quên" với khoảng bỏ quên thật.
   - Recall: tỉ lệ khoảng bỏ quên thật được cảnh báo; Precision: tỉ lệ cảnh báo đúng
   - Độ trễ cảnh báo (tính từ lúc người cuối cùng rời phòng, và phần vượt Te)
   - Số cảnh báo giả / phòng / ngày
2) Mức thành phần theo khung (tuỳ chọn --frame-log + --frame-gt):
   - Độ chính xác xác định phòng trống; Precision/Recall/F1 trạng thái từng thiết bị.

ground_truth.csv:  room_id,start_s,end_s[,devices,note]   (start/end: giây kể từ --start, hoặc ISO datetime)
frame_gt.csv:      entity,start_s,end_s                    (entity = person | id thiết bị; khoảng đang có người/BẬT)

    python scripts/evaluate_events.py --events runs/demo_events.jsonl --gt data/demo/ground_truth.csv \
        --start 2026-10-01T17:00:00 --frame-log runs/demo_frames.jsonl --frame-gt data/demo/frame_gt.csv --te 20
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime


def _t(v: str, t0: float) -> float:
    try:
        return t0 + float(v)
    except ValueError:
        return datetime.fromisoformat(v).timestamp()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def event_level(events: list[dict], gt_rows: list[dict], t0: float, tol: float, days: float, te: float | None) -> dict:
    alerts = [e for e in events if e["type"] == "alert"]
    closed = {e["alert_id"]: e for e in events if e["type"] == "alert_closed"}
    gts = [(r["room_id"], _t(r["start_s"], t0), _t(r["end_s"], t0)) for r in gt_rows]
    matched_gt, tp_alerts, latencies = set(), 0, []
    for a in alerts:
        hit = [i for i, (room, s, e) in enumerate(gts) if room == a["room_id"] and s <= a["ts"] <= e + tol]
        if hit:
            tp_alerts += 1
            i = hit[0]
            if i not in matched_gt:
                matched_gt.add(i)
                latencies.append(a["ts"] - gts[i][1])
    fp = len(alerts) - tp_alerts
    fn = len(gts) - len(matched_gt)
    p, r, f = prf(tp_alerts, fp, fn)
    rooms = max(1, len({a["room_id"] for a in alerts} | {g[0] for g in gts}))
    res = {
        "gt_intervals": len(gts), "alerts": len(alerts), "true_alerts": tp_alerts, "false_alerts": fp, "missed": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "false_alerts_per_room_day": round(fp / rooms / max(days, 1e-9), 3),
        "latency_mean_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "latency_max_s": round(max(latencies), 2) if latencies else None,
        "waste_wh_detected": round(sum(c.get("waste_wh", 0.0) for c in closed.values()), 3),
    }
    if te is not None and latencies:
        res["latency_over_te_mean_s"] = round(sum(latencies) / len(latencies) - te, 2)
    return res


def frame_level(frame_log: str, frame_gt: str, t0: float, margin: float) -> dict:
    intervals = defaultdict(list)
    with open(frame_gt, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            intervals[r["entity"]].append((_t(r["start_s"], t0), _t(r["end_s"], t0)))

    def truth(entity, ts):
        iv = intervals.get(entity, [])
        if any(abs(ts - s) < margin or abs(ts - e) < margin for s, e in iv):
            return None  # bỏ qua khung sát thời điểm chuyển trạng thái
        return any(s <= ts < e for s, e in iv)

    empty_ok = empty_n = 0
    dev = defaultdict(lambda: [0, 0, 0, 0])   # tp fp fn tn
    with open(frame_log, encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            ts = fr["ts"]
            gt_person = truth("person", ts)
            if gt_person is not None:
                empty_n += 1
                empty_ok += int((fr["n"] == 0) == (not gt_person))
            for d, st in fr["devices"].items():
                if d not in intervals:
                    continue
                g = truth(d, ts)
                if g is None or st == "UNKNOWN":
                    continue
                pred = st == "ON"
                dev[d][0 if pred and g else 1 if pred and not g else 2 if g else 3] += 1
    out = {"empty_room_accuracy": round(empty_ok / empty_n, 4) if empty_n else None, "frames": empty_n, "devices": {}}
    for d, (tp, fp, fn, tn) in sorted(dev.items()):
        p, r, f1 = prf(tp, fp, fn)
        out["devices"][d] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                             "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="+", required=True, help="một hoặc nhiều file JSONL (so sánh cấu hình/ablation)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--start", help="thời điểm bắt đầu video (ISO) nếu GT dùng giây tương đối")
    ap.add_argument("--tol", type=float, default=30.0, help="cho phép cảnh báo muộn sau khi khoảng bỏ quên kết thúc (s)")
    ap.add_argument("--days", type=float, help="số ngày quan sát (mặc định suy ra từ dữ liệu, tối thiểu 1)")
    ap.add_argument("--te", type=float, help="Te đã cấu hình, để báo độ trễ vượt Te")
    ap.add_argument("--frame-log")
    ap.add_argument("--frame-gt")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    t0 = datetime.fromisoformat(a.start).timestamp() if a.start else 0.0
    with open(a.gt, encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    report = {}
    for path in a.events:
        with open(path, encoding="utf-8") as f:
            events = [json.loads(l) for l in f if l.strip()]
        span = (max(e["ts"] for e in events) - min(e["ts"] for e in events)) / 86400 if events else 0
        days = a.days or max(1.0, span)
        report[path] = event_level(events, gt_rows, t0, a.tol, days, a.te)

    if a.frame_log and a.frame_gt:
        report["frame_level"] = frame_level(a.frame_log, a.frame_gt, t0, a.margin)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
