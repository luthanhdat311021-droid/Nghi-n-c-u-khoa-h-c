"""Huấn luyện bộ phân lớp MobileNetV3-Small nhận diện BẬT/TẮT trên ảnh ROI của một loại thiết bị.

Dữ liệu (sinh tự động bởi ecoclass.data.autolabel):
    datasets/roi_cls/<type>/train/{off,on}/*.jpg
    datasets/roi_cls/<type>/val/{off,on}/*.jpg

    python scripts/train_device_cls.py --type light --epochs 15
Đầu ra: models/<type>_mnv3.onnx (dùng trong cấu hình: classifier: models/light_mnv3.onnx)
và báo cáo Precision/Recall/F1 trên tập val.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import torch
    from sklearn.metrics import classification_report, f1_score
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True)
    ap.add_argument("--data", default="datasets/roi_cls")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    root = Path(a.data) / a.type
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Không lật/xoay mạnh: ROI cố định. Tăng cường độ sáng/tương phản để chống thay đổi ánh sáng tự nhiên.
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    tf_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    ds_tr = datasets.ImageFolder(root / "train", tf_train)
    ds_va = datasets.ImageFolder(root / "val", tf_val)
    assert ds_tr.classes == ["off", "on"], f"Cần đúng 2 thư mục off/on, có: {ds_tr.classes}"

    # Cân bằng lớp bằng trọng số loss
    counts = torch.bincount(torch.tensor(ds_tr.targets), minlength=2).float()
    weights = (counts.sum() / (2 * counts.clamp(min=1))).to(dev)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    dl_tr = DataLoader(ds_tr, a.batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, a.batch, num_workers=2)

    def evaluate():
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in dl_va:
                ps += model(x.to(dev)).argmax(1).cpu().tolist()
                ys += y.tolist()
        return ys, ps

    best, out = -1.0, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        ys, ps = evaluate()
        f1 = f1_score(ys, ps, pos_label=1)
        print(f"epoch {ep + 1}/{a.epochs} loss={tot / len(ds_tr):.4f} val F1(on)={f1:.4f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out / f"{a.type}_mnv3.pth")

    model.load_state_dict(torch.load(out / f"{a.type}_mnv3.pth", map_location=dev))
    ys, ps = evaluate()
    print(classification_report(ys, ps, target_names=["off", "on"], digits=4))

    model.cpu().eval()
    onnx_path = out / f"{a.type}_mnv3.onnx"
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_path, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    print(f"[train] best F1={best:.4f} → {onnx_path}")


if __name__ == "__main__":
    main()
