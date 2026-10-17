"""Tải dữ liệu công khai Open Images V7 (Google, CC BY 4.0) cho các lớp thiết bị → định dạng YOLO.

Lớp hệ thống ← lớp Open Images:
    fan    ← Ceiling fan (/m/03ldnb), Mechanical fan (/m/02x984l)
    window ← Window (/m/0d4v4)            (để loại cửa sổ sáng khỏi bộ dò đèn)
File chú thích train (~2,3 GB) được đọc dạng luồng và lọc ngay, không lưu xuống đĩa.
Ảnh được thu nhỏ còn cạnh dài 640 px trước khi lưu.

    python scripts/fetch_openimages.py --out datasets/oi_devices --max-per-class 1500
    # thêm ảnh âm (không có thiết bị, nhãn rỗng) từ phiên ghi webcam:
    python scripts/fetch_openimages.py --out datasets/oi_devices --negatives datasets/raw/session1 --neg-count 150
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

BASE = "https://storage.googleapis.com/openimages"
ANN = {
    "train": f"{BASE}/v6/oidv6-train-annotations-bbox.csv",
    "validation": f"{BASE}/v5/validation-annotations-bbox.csv",
    "test": f"{BASE}/v5/test-annotations-bbox.csv",
}
IMG_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{iid}.jpg"
CLASSES = {"fan": ["/m/03ldnb", "/m/02x984l"], "window": ["/m/0d4v4"]}
NAMES = list(CLASSES)


def stream_annotations(url: str, mids: dict[str, int]) -> dict[str, list]:
    """ImageID -> [(cls, xmin, ymin, xmax, ymax)], đọc luồng CSV và lọc theo lớp."""
    out: dict[str, list] = defaultdict(list)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        lines = r.iter_lines(decode_unicode=True)
        header = next(lines).split(",")
        idx = {k: header.index(k) for k in ("ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf")}
        n = 0
        for line in lines:
            n += 1
            if n % 2_000_000 == 0:
                print(f"  … {n / 1e6:.0f}M dòng", flush=True)
            parts = line.split(",")
            c = mids.get(parts[idx["LabelName"]])
            if c is None:
                continue
            out[parts[idx["ImageID"]]].append((c, float(parts[idx["XMin"]]), float(parts[idx["YMin"]]),
                                               float(parts[idx["XMax"]]), float(parts[idx["YMax"]])))
    return out


def fetch(job) -> bool:
    split, iid, boxes, dst_img, dst_lbl = job
    if dst_img.exists():
        return True
    try:
        r = requests.get(IMG_URL.format(split=split, iid=iid), timeout=60)
        if r.status_code != 200:
            return False
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        s = 640 / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        dst_lbl.write_text("".join(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n"
                                   for c, x1, y1, x2, y2 in boxes))
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="datasets/oi_devices")
    ap.add_argument("--max-per-class", type=int, default=1500, help="số ảnh tối đa mỗi lớp (tập train)")
    ap.add_argument("--negatives", nargs="*", default=[], help="thư mục ảnh không có quạt/cửa sổ → nhãn rỗng")
    ap.add_argument("--neg-count", type=int, default=150)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true", help="chỉ thêm ảnh âm")
    a = ap.parse_args()

    out = Path(a.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    mids = {m: i for i, (k, ms) in enumerate(CLASSES.items()) for m in ms}
    rnd = random.Random(0)

    if not a.skip_download:
        jobs = []
        for oi_split, sp in (("validation", "val"), ("test", "val"), ("train", "train")):
            print(f"[oi] đọc chú thích {oi_split} …", flush=True)
            ann = stream_annotations(ANN[oi_split], mids)
            ids = list(ann)
            rnd.shuffle(ids)
            cap = a.max_per_class if sp == "train" else max(150, a.max_per_class // 6)
            # Ưu tiên ảnh có quạt (lớp hiếm), cửa sổ chỉ lấy tới giới hạn.
            per = defaultdict(int)
            for iid in sorted(ids, key=lambda i: 0 if any(b[0] == 0 for b in ann[i]) else 1):
                classes = {b[0] for b in ann[iid]}
                if all(per[c] >= cap for c in classes):
                    continue
                for c in classes:
                    per[c] += 1
                jobs.append((oi_split, iid, ann[iid], out / "images" / sp / f"oi_{iid}.jpg",
                             out / "labels" / sp / f"oi_{iid}.txt"))
            print(f"[oi] {oi_split}: chọn {dict((NAMES[k], v) for k, v in per.items())}", flush=True)

        print(f"[oi] tải {len(jobs)} ảnh …", flush=True)
        ok = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for i, res in enumerate(ex.map(fetch, jobs), 1):
                ok += res
                if i % 250 == 0:
                    print(f"  … {i}/{len(jobs)} (thành công {ok})", flush=True)
        print(f"[oi] tải xong {ok}/{len(jobs)}")

    for d in a.negatives:
        files = sorted(Path(d).glob("*.jpg"))
        for f in rnd.sample(files, min(a.neg_count, len(files))):
            sp = "val" if rnd.random() < 0.15 else "train"
            img = cv2.imread(str(f))
            cv2.imwrite(str(out / "images" / sp / f"neg_{Path(d).name}_{f.stem}.jpg"), img)
            (out / "labels" / sp / f"neg_{Path(d).name}_{f.stem}.txt").write_text("")
        print(f"[oi] thêm ảnh âm từ {d}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    for sp in ("train", "val"):
        print(f"[oi] {sp}: {len(list((out / 'images' / sp).glob('*.jpg')))} ảnh")


if __name__ == "__main__":
    main()
