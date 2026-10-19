"""Bộ phân lớp chạy GPU (CUDA Graph) phải cho cùng kết quả với bản ONNX, kể cả khi nhiều bộ cùng tồn tại.

Lỗi cũ: tensor mean/std là biến cục bộ → bị giải phóng, bộ phân lớp tạo sau ghi đè vùng nhớ → đèn 0,77 thành 0,38.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
if not torch.cuda.is_available():
    pytest.skip("cần GPU NVIDIA", allow_module_level=True)

from ecoclass.edge.devices import ROIClassifier  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"
LIGHT, FAN = MODELS / "light_scene_mnv3.onnx", MODELS / "fan_scene_mnv3.onnx"
if not (LIGHT.exists() and LIGHT.with_suffix(".pth").exists() and FAN.with_suffix(".pth").exists()):
    pytest.skip("thiếu mô hình", allow_module_level=True)


def _onnx_p(path, clf, img):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    lo = sess.run(None, {sess.get_inputs()[0].name: clf.preprocess(img)})[0][0]
    e = np.exp(lo - lo.max())
    return float(e[1] / e.sum())


def test_gpu_matches_onnx_with_several_classifiers():
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    light = ROIClassifier(str(LIGHT), [384, 224])
    assert light.gpu_model is not None
    expected = [_onnx_p(LIGHT, light, im) for im in imgs]
    fan = ROIClassifier(str(FAN), [384, 224])          # tạo bộ thứ hai rồi chạy xen kẽ
    for im, exp in zip(imgs, expected):
        fan.prob_on(im)
        torch.empty(10_000_000, device="cuda").fill_(123.0)   # cấp/ghi vùng nhớ mới như YOLO
        assert abs(light.prob_on(im) - exp) < 2e-3
