"""Huấn luyện bộ phân lớp BẬT/TẮT toàn khung (MobileNetV3) cho một loại thiết bị từ datasets/scene_cls.

    python scripts/train_scene_cls.py --task light --input rgb
    python scripts/train_scene_cls.py --task fan --input tstack
Đánh giá trên các video val (chia theo video, xem build_scene_cls.py): độ chính xác/F1 mức khung và
mức video (đa số khung). Khi bộ dữ liệu không có val (``--val none`` lúc tạo) → huấn luyện bản cuối
trên toàn bộ dữ liệu với số epoch cố định.
Huấn luyện tiếp (fine-tune) từ mô hình đã có trên dữ liệu cũ + dữ liệu mới, ghi ra thư mục RIÊNG:
    python scripts/train_scene_cls.py --task light --data datasets/scene_cls_all datasets/scene_cls_imou
        --init models/light_scene_mnv3.pth --lr 2e-4 --out models/imou
Đầu ra: models/<task>_scene_mnv3.onnx (lớp 0 = off, 1 = on), dùng trong cấu hình thiết bị:
    classifier: ../models/fan_scene_mnv3.onnx
    params: {cls_input: tstack, cls_size: [384, 224]}
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import f1_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["light", "fan"])
    ap.add_argument("--input", choices=["rgb", "tstack"], help="mặc định: light→rgb, fan→tstack")
    ap.add_argument("--data", nargs="+", default=["datasets/scene_cls"], help="một hoặc nhiều thư mục dữ liệu (gộp lại)")
    ap.add_argument("--init", help="trọng số .pth để huấn luyện tiếp (chỉ đọc); mặc định: ImageNet")
    ap.add_argument("--arch", default="small", choices=["small", "large"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inp = a.input or ("rgb" if a.task == "light" else "tstack")
    torch.manual_seed(a.seed)

    rows = []
    for d in a.data:
        for r in csv.DictReader(open(Path(d) / "index.csv", encoding="utf-8")):
            if r.get(a.task, "") != "":           # bỏ mẫu không có nhãn cho thiết bị này
                rows.append({**r, "root": d})
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter = [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)] if inp == "rgb" \
        else [transforms.ColorJitter(brightness=0.3, contrast=0.3)]   # tstack: không đổi màu (màu = chuyển động)
    tf_train = transforms.Compose([
        transforms.RandomAffine(degrees=4, translate=(0.06, 0.06), scale=(0.9, 1.15)),   # camera xê dịch/nghiêng
        *jitter, transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    tf_val = transforms.Compose([transforms.ToTensor(), norm])

    class DS(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            r = self.items[i]
            img = Image.open(Path(r["root"]) / inp / r["file"]).convert("RGB")
            return self.tf(img), int(r[a.task])

    def build():
        if a.arch == "small":
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        return m.to(dev)

    ys_tr = torch.tensor([int(r[a.task]) for r in tr])
    counts = torch.bincount(ys_tr, minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts.clamp(min=1))).to(dev), label_smoothing=0.05)
    dl_tr = DataLoader(DS(tr, tf_train), a.batch, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(DS(va, tf_val), a.batch, num_workers=0) if va else None

    model = build()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
        print(f"[train] huấn luyện tiếp từ {a.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl_tr), pct_start=0.15)

    def predict(dl):
        model.eval()
        ps = []
        with torch.no_grad():
            for x, _ in dl:
                ps += torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().tolist()
        return np.array(ps)

    def report(p, items, tag=""):
        y = np.array([int(r[a.task]) for r in items])
        pred = (p >= 0.5).astype(int)
        acc = float((pred == y).mean())
        f1 = f1_score(y, pred, average="macro")
        per_vid = defaultdict(list)
        for r, q in zip(items, pred):
            per_vid[r["video"]].append(q == int(r[a.task]))
        vid = {k: float(np.mean(v)) for k, v in per_vid.items()}
        print(f"{tag}acc={acc:.4f} macroF1={f1:.4f} | " + " ".join(f"{k}:{v:.2f}" for k, v in sorted(vid.items())))
        return f1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pth = out / f"{a.task}_scene_mnv3.pth"
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        msg = f"epoch {ep + 1}/{a.epochs} loss={tot / (len(dl_tr) * a.batch):.4f} "
        if dl_va is not None:
            f1 = report(predict(dl_va), va, msg + "val ")
            # Giữ epoch tốt nhất; hoà → lấy epoch sau (đã học lâu hơn với LR nhỏ hơn).
            if f1 >= best:
                best = f1
                torch.save(model.state_dict(), pth)
        else:
            print(msg, flush=True)
            torch.save(model.state_dict(), pth)

    model.load_state_dict(torch.load(pth, map_location=dev))
    if dl_va is not None:
        report(predict(dl_va), va, "[best] val ")
    model.cpu().eval()
    onnx_path = out / f"{a.task}_scene_mnv3.onnx"
    w, h = 384, 224
    torch.onnx.export(model, torch.randn(1, 3, h, w), onnx_path, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    print(f"[train] input={inp} → {onnx_path}")


if __name__ == "__main__":
    main()
