import cv2
import numpy as np

from ecoclass.edge.tstack import make_tstack


def _scene(shift: int = 0, blade: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    g = cv2.GaussianBlur(rng.integers(0, 255, (224, 384)).astype(np.uint8), (0, 0), 1.5)   # kết cấu mượt như ảnh thật
    g = cv2.normalize(g, None, 40, 220, cv2.NORM_MINMAX)
    g = np.roll(g, shift, axis=1)
    if blade is not None:
        g[20:30, 150 + blade:190 + blade] = 255   # cánh quạt nhỏ so với cả khung (như ảnh thật)
    return g


def test_camera_shift_is_aligned_away():
    """Camera tịnh tiến (rung tay) → sau căn chỉnh ba kênh gần như bằng nhau (không có "chuyển động")."""
    st = make_tstack(_scene(0), _scene(3), _scene(6)).astype(int)
    inner = st[10:-10, 20:-20]
    raw = np.abs(_scene(0).astype(int) - _scene(6))[10:-10, 20:-20].mean()
    assert np.abs(inner[..., 0] - inner[..., 2]).mean() < 0.2 * raw


def test_moving_blade_differs_between_channels():
    st = make_tstack(_scene(blade=0), _scene(blade=60), _scene(blade=120)).astype(int)
    assert np.abs(st[20:30, 150:190, 0] - st[20:30, 150:190, 2]).mean() > 50


def test_person_mask_removes_motion():
    mask = np.zeros((224, 384), bool)
    mask[10:40, 140:320] = True
    st = make_tstack(_scene(blade=0), _scene(blade=60), _scene(blade=120), mask).astype(int)
    assert np.abs(st[..., 0] - st[..., 2])[mask].max() == 0
