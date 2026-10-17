"""Nhận diện trạng thái BẬT/TẮT của từng thiết bị trên vùng quan sát (ROI) — Bảng 2.

| Thiết bị          | Dấu hiệu                         | Kỹ thuật                                                  |
|-------------------|----------------------------------|-----------------------------------------------------------|
| light             | bóng sáng, quầng sáng            | tỉ số độ sáng kênh V (HSV) ROI / vùng nền                 |
| lights_auto       | các vùng sáng bão hoà, màu trắng | tìm đèn đang sáng trên toàn khung (không cần ROI cố định)  |
| fan               | cánh quay, nhòe chuyển động      | dao động độ sáng theo thời gian + Farnebäck + độ nét       |
| screen / monitor  | màn sáng, có nội dung            | độ sáng tương đối + phương sai kết cấu                    |
| ac                | đèn báo dàn lạnh sáng            | tỉ lệ điểm ảnh sáng có màu trong ROI nhỏ                  |

- Mọi thiết bị đều có thể gắn bộ phân lớp MobileNetV3 (``classifier``) thay cho luật thủ công.
- ``params.track: true`` → ROI dịch theo chuyển động camera (ước lượng bằng tương quan pha, motion.py).
- Che khuất được tính bằng mặt nạ phân đoạn người (YOLO11-seg) nếu có, ngược lại dùng box người.
- Trạng thái cuối cùng có trễ: cần ``on_frames``/``off_frames`` lần đọc liên tiếp mới lật trạng thái.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import DeviceConfig
from ..fsm import ON, OFF, UNKNOWN
from .geometry import crop, overlap_ratio, roi_to_px


@dataclass
class DeviceReading:
    state: str                  # trạng thái đã làm trễ
    raw: str                    # kết quả đo của khung hiện tại
    score: float
    features: dict = field(default_factory=dict)


@dataclass
class SceneInfo:
    """Thông tin dùng chung cho mọi thiết bị trong một khung hình."""
    hsv: np.ndarray
    gray: np.ndarray
    mean_v: float
    dark: bool
    person_boxes: list
    person_mask: np.ndarray | None = None   # bool HxW, True = điểm ảnh thuộc người
    cam_offset: tuple[float, float] = (0.0, 0.0)   # dịch tích luỹ của camera (pixel) so với lúc cấu hình ROI
    cam_moving: bool = False
    cam_reliable: bool = True          # False: khung không còn khớp tham chiếu (camera quay đi hướng khác)
    device_boxes: dict | None = None   # {"light": [...], "fan": [...]} từ DeviceLocator (YOLO), toạ độ chuẩn hoá


def make_scene(frame: np.ndarray, person_boxes: list, person_mask: np.ndarray | None = None,
               motion=None, dark_v: float = 25.0, device_boxes: dict | None = None) -> SceneInfo:
    """``motion``: CameraMotion (tuỳ chọn) — cập nhật độ dịch camera cho các ROI có track."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_v = float(hsv[..., 2].mean())
    ox = oy = 0.0
    moving, reliable = False, True
    if motion is not None:
        ox, oy, moving = motion.update(gray, person_mask)
        reliable = motion.reliable
    return SceneInfo(hsv=hsv, gray=gray, mean_v=mean_v, dark=mean_v < dark_v, person_boxes=person_boxes,
                     person_mask=person_mask, cam_offset=(ox, oy), cam_moving=moving, cam_reliable=reliable,
                     device_boxes=device_boxes)


def _overlap(a, b) -> float:
    """Tỉ lệ diện tích box a (chuẩn hoá) nằm trong box b."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return iw * ih / area


def _box_px(b, w, h):
    return int(b[0] * w), int(b[1] * h), max(int(b[2] * w), int(b[0] * w) + 1), max(int(b[3] * h), int(b[1] * h) + 1)


def shift_roi(roi: list[float], dx: float, dy: float) -> list[float] | None:
    """Dịch ROI chuẩn hoá; cắt theo mép khung. None nếu phần còn lại quá nhỏ (thiết bị ra ngoài khung)."""
    x1, y1, x2, y2 = roi[0] + dx, roi[1] + dy, roi[2] + dx, roi[3] + dy
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if (cx2 - cx1) < 0.5 * (x2 - x1) or (cy2 - cy1) < 0.5 * (y2 - y1):
        return None
    return [round(cx1, 4), round(cy1, 4), round(cx2, 4), round(cy2, 4)]


class DeviceDetector:
    occlusion_max = 0.3
    # Khi ROI bị người che, chỉ tin kết quả đo mà sự che khuất KHÔNG thể tạo ra:
    # người đứng trước đèn/màn hình chỉ làm vùng tối đi (→ OFF giả), không tạo ra đèn sáng.
    trust_when_occluded = ON

    def __init__(self, cfg: DeviceConfig):
        self.cfg = cfg
        self.p = cfg.params
        self.state = UNKNOWN
        self._streak_state = UNKNOWN
        self._streak = 0
        self.roi = list(cfg.roi)                                  # ROI hiện tại (có thể đã dịch)
        self.ref_roi = list(cfg.ref_roi) if cfg.ref_roi else None
        self.camera_moved = False
        self.classifier = ROIClassifier(cfg.classifier, size=self.p.get("cls_size", 224)) if cfg.classifier else None
        # "roi": ảnh màu của ROI; "tstack": ảnh chồng thời gian toàn khung (tstack.py) — cho quạt.
        self.cls_input = self.p.get("cls_input", "roi")
        self._hist: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        self.track = bool(self.p.get("track"))

    # Các lớp con cài đặt measure(): trả về (raw_state, score, features)
    def measure(self, frame: np.ndarray, scene: SceneInfo, ts: float) -> tuple[str, float, dict]:
        raise NotImplementedError

    def update(self, frame: np.ndarray, scene: SceneInfo, ts: float) -> DeviceReading:
        h, w = frame.shape[:2]
        self.camera_moved = False
        if self.track:
            dx, dy = scene.cam_offset[0] / w, scene.cam_offset[1] / h
            roi = shift_roi(self.cfg.roi, dx, dy) if scene.cam_reliable else None
            sim = self._appearance(scene.gray, roi) if roi is not None else None
            if roi is None or (sim is not None and sim < self.p.get("min_similarity", 0.3)):
                return DeviceReading(self.state, UNKNOWN, 0.0,
                                     {"out_of_view": True, "similarity": None if sim is None else round(sim, 2)})
            self.roi = roi
            self.ref_roi = shift_roi(self.cfg.ref_roi, dx, dy) if self.cfg.ref_roi else None
            self.camera_moved = scene.cam_moving

        occ = self.occlusion(scene, w, h)
        if self.classifier is not None and not scene.dark:
            x = self._tstack_input(scene, ts) if self.cls_input == "tstack" else crop(frame, self.roi)
            if x is None:
                return DeviceReading(self.state, UNKNOWN, 0.0, {"warming_up": True})
            p_on = self.classifier.prob_on(x)
            raw, score, feats = (ON if p_on >= self.p.get("cls_threshold", 0.5) else OFF), p_on, {"p_on": round(p_on, 3)}
        else:
            raw, score, feats = self.measure(frame, scene, ts)
        feats["occlusion"] = round(occ, 2)

        if occ > self.occlusion_max and raw != self.trust_when_occluded:
            return DeviceReading(self.state, UNKNOWN, float(score), {**feats, "occluded": True})
        self._apply_hysteresis(raw)
        return DeviceReading(self.state, raw, float(score), feats)

    def _tstack_input(self, scene: SceneInfo, ts: float) -> np.ndarray | None:
        """Ảnh chồng thời gian [t-2dt, t-dt, t] từ lịch sử khung; None khi chưa đủ lịch sử.
        Cần pipeline chạy >= 1/dt fps (mặc định dt = 0,1 s → 10 fps) để khớp dữ liệu huấn luyện."""
        from .tstack import make_tstack, to_gray_small
        size = tuple(self.classifier.size)
        g0 = to_gray_small(scene.gray, size)
        self._hist.append((ts, g0))
        dt = float(self.p.get("cls_dt", 0.1))
        past = list(self._hist)[:-1]
        if len(past) < 2:
            return None
        pick = [min(past, key=lambda h: abs(h[0] - (ts - k * dt))) for k in (2, 1)]
        if pick[0][0] == pick[1][0] or abs(pick[1][0] - (ts - dt)) > dt:
            return None
        return make_tstack(pick[0][1], pick[1][1], g0, scene.person_mask)

    def _appearance(self, gray: np.ndarray, roi: list[float]) -> float | None:
        """Tương quan chuẩn hoá giữa ROI hiện tại và ROI ở khung tham chiếu (ảnh làm mờ mạnh,
        để cánh quạt quay không ảnh hưởng). Khung đầu tiên: lưu mẫu, trả về None."""
        patch = cv2.GaussianBlur(cv2.resize(crop(gray, roi), (48, 48), interpolation=cv2.INTER_AREA), (0, 0), 3)
        patch = patch.astype(np.float32)
        if getattr(self, "_ref_patch", None) is None:
            self._ref_patch = patch
            return None
        a, b = self._ref_patch - self._ref_patch.mean(), patch - patch.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 1e-6 else 0.0

    def occlusion(self, scene: SceneInfo, w: int, h: int) -> float:
        if scene.person_mask is not None:
            m = crop(scene.person_mask, self.roi)
            return float(m.mean()) if m.size else 0.0
        roi_px = roi_to_px(self.roi, w, h)
        return max((overlap_ratio(roi_px, b[:4]) for b in scene.person_boxes), default=0.0)

    def _apply_hysteresis(self, raw: str) -> None:
        if raw == UNKNOWN:
            return
        if raw == self._streak_state:
            self._streak += 1
        else:
            self._streak_state, self._streak = raw, 1
        need = self.cfg.on_frames if raw == ON else self.cfg.off_frames
        if self.state == UNKNOWN or self._streak >= need:
            self.state = raw

    # tiện ích
    def _v(self, scene: SceneInfo, roi) -> np.ndarray:
        return crop(scene.hsv, roi)[..., 2]

    def _ref_v(self, scene: SceneInfo) -> float:
        if self.ref_roi:
            return float(self._v(scene, self.ref_roi).mean())
        return scene.mean_v

    def _person_mask(self, scene: SceneInfo, shape: tuple[int, int] | None = None) -> np.ndarray | None:
        if scene.person_mask is None:
            return None
        m = crop(scene.person_mask, self.roi).astype(np.uint8)
        if shape is not None:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)


class LightDetector(DeviceDetector):
    """Đèn: tỉ số độ sáng V(ROI)/V(nền) + tỉ lệ điểm ảnh gần bão hoà (quầng sáng)."""

    def measure(self, frame, scene, ts):
        v = self._v(scene, self.roi)
        v_roi = float(v.mean())
        v_ref = self._ref_v(scene)
        ratio = v_roi / (v_ref + 1.0)
        sat = float((v >= self.p.get("sat_level", 235)).mean())
        thr = self._ratio_threshold()
        feats = {"ratio": round(ratio, 3), "sat_frac": round(sat, 3), "v_roi": round(v_roi, 1),
                 "v_ref": round(v_ref, 1), "thr": round(thr, 3)}
        if scene.dark and v_roi < 60:
            return OFF, ratio, feats
        on = ratio >= thr or sat >= self.p.get("sat_frac_on", 0.08)
        return (ON if on else OFF), ratio, feats

    def _ratio_threshold(self) -> float:
        # Sau khi hiệu chuẩn (scripts/calibrate_devices.py) ngưỡng = trung điểm hai mức đo.
        if "ref_on_ratio" in self.p and "ref_off_ratio" in self.p:
            return 0.5 * (self.p["ref_on_ratio"] + self.p["ref_off_ratio"])
        return self.p.get("on_ratio", 1.35)


class LightGroupDetector(DeviceDetector):
    """Tìm các đèn đang sáng trên toàn khung hình (hoặc trong ROI lớn), không cần ROI từng đèn.

    Đèn huỳnh quang/LED đang bật là vùng gần bão hoà (V cao), ít màu (S thấp), có diện tích
    vừa phải và khá đặc. Cách này không bị ảnh hưởng khi camera xê dịch; đổi lại chỉ biết
    *số đèn đang sáng trong khung*, không phân biệt từng đèn. Công suất = số đèn × per_light_w.
    """

    occlusion_max = 1.1   # không coi cả khung là bị che; điểm ảnh người bị loại trong measure()

    def __init__(self, cfg):
        super().__init__(cfg)
        self.blobs: list[list[float]] = []
        self.dynamic_power: float | None = None

    def measure(self, frame, scene, ts):
        if scene.device_boxes is not None and "light" in scene.device_boxes:
            return self._measure_in_boxes(scene)
        hsv = crop(scene.hsv, self.roi)
        H, W = hsv.shape[:2]
        # Chỉ ngưỡng theo độ sáng; độ bão hoà màu được xét theo trung bình cả vùng ở dưới
        # (lọc từng điểm ảnh sẽ cắt vụn ô cửa sổ thành mảnh nhỏ trông giống đèn).
        m = (hsv[..., 2] >= self.p.get("v_min", 240)) & (hsv[..., 1] <= self.p.get("s_max", 255))
        pm = self._person_mask(scene)
        if pm is not None:
            m &= ~pm
        m = m.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        area = H * W
        rx1, ry1, rx2, ry2 = self.roi
        sat = hsv[..., 1].astype(np.float32)
        # Trung bình S của từng thành phần (vector hoá).
        sums = np.bincount(lab.ravel(), weights=sat.ravel(), minlength=n)
        cnts = np.bincount(lab.ravel(), minlength=n).astype(np.float32)
        mean_s = sums / np.maximum(cnts, 1)
        core = (hsv[..., 2] >= self.p.get("core_v", 250)).astype(np.float32)
        core_frac = np.bincount(lab.ravel(), weights=core.ravel(), minlength=n) / np.maximum(cnts, 1)
        min_nb_area = self.p.get("neighbor_min_area", 30)
        blobs = []
        for i in range(1, n):
            x, y, bw, bh, a = stats[i]
            # Ánh sáng đèn huỳnh quang/LED gần như không màu (S ~2–3); ô cửa sổ nhìn ra trời có ánh xanh.
            if mean_s[i] > self.p.get("s_mean_max", 8):
                continue
            # Cửa sổ = lưới nhiều ô sáng sát nhau, ngăn bởi khung mảnh; máng đèn đứng riêng lẻ.
            gap = self.p.get("neighbor_gap", 0.3) * min(bw, bh) + 4
            nb = 0
            for j in range(1, n):
                if j == i or stats[j][4] < min_nb_area:
                    continue
                X, Y, Wj, Hj, _ = stats[j]
                gx = max(0, max(x, X) - min(x + bw, X + Wj))
                gy = max(0, max(y, Y) - min(y + bh, Y + Hj))
                if gx < gap and gy < gap:
                    nb += 1
            if nb > self.p.get("max_neighbors", 0):
                continue
            if not (self.p.get("min_area", 0.002) * area <= a <= self.p.get("max_area", 0.12) * area):
                continue
            if a / float(bw * bh) < self.p.get("fill_min", 0.4):
                continue
            # Mọi loại đèn (tuýp, tròn âm trần, panel, bóng tròn) đều là NGUỒN SÁNG: lõi gần như toàn bộ
            # cháy trắng (V>=250). Tường/cửa sổ cháy sáng do tự phơi sáng chỉ "gần trắng" (tỉ lệ thấp hơn).
            # Không lọc theo hình dạng → nhận cả đèn tròn/vuông.
            if core_frac[i] < self.p.get("core_min", 0.88):
                continue
            if max(bw, bh) / float(min(bw, bh)) < self.p.get("min_aspect", 0.0):
                continue
            blobs.append([round(rx1 + (rx2 - rx1) * x / W, 4), round(ry1 + (ry2 - ry1) * y / H, 4),
                          round(rx1 + (rx2 - rx1) * (x + bw) / W, 4), round(ry1 + (ry2 - ry1) * (y + bh) / H, 4)])
        if scene.dark:
            blobs = []
        # Có bộ phát hiện YOLO lớp "window": vùng sáng nằm trong cửa sổ không phải đèn.
        windows = (scene.device_boxes or {}).get("window", [])
        if windows:
            blobs = [b for b in blobs if max((_overlap(b, wb) for wb in windows), default=0.0)
                     < self.p.get("window_overlap_max", 0.3)]
        self.blobs = blobs
        self.dynamic_power = len(blobs) * float(self.p.get("per_light_w", 72))
        return (ON if blobs else OFF), float(len(blobs)), {"count": len(blobs), "windows": len(windows)}

    def _measure_in_boxes(self, scene):
        """Có bộ phát hiện YOLO: chỉ xét các máng đèn nó tìm thấy; đèn BẬT nếu đủ điểm ảnh bão hoà."""
        H, W = scene.hsv.shape[:2]
        v = scene.hsv[..., 2]
        blobs = []
        for b in scene.device_boxes["light"]:
            x1, y1, x2, y2 = _box_px(b, W, H)
            if (v[y1:y2, x1:x2] >= self.p.get("v_min", 240)).mean() >= self.p.get("box_bright_frac", 0.25):
                blobs.append(b)
        self.blobs = [] if scene.dark else blobs
        self.dynamic_power = len(self.blobs) * float(self.p.get("per_light_w", 72))
        return (ON if self.blobs else OFF), float(len(self.blobs)), \
            {"count": len(self.blobs), "fixtures": len(scene.device_boxes["light"])}


class FanDetector(DeviceDetector):
    """Quạt: kết hợp ba đặc trưng trong ROI cánh quạt (đã loại điểm ảnh thuộc người):

    1. Dao động độ sáng theo thời gian: tỉ lệ điểm ảnh có độ lệch chuẩn theo thời gian lớn
       trên cửa sổ vài khung. Hiệu quả với cánh quạt trắng trên trần trắng (tương phản thấp),
       nơi dòng quang học yếu.
    2. Năng lượng dòng quang học Farnebäck, ngưỡng thích nghi theo nhiễu nền.
    3. Độ nét (phương sai Laplacian): cánh quay nhanh bị nhòe thành đĩa mờ.
    Khi camera đang xê dịch (ROI dịch > move_px) các đặc trưng chuyển động không tin cậy → UNKNOWN.
    """

    trust_when_occluded = OFF   # người cử động trong ROI tạo chuyển động giả (→ ON giả)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prev: np.ndarray | None = None
        self.energies: deque[float] = deque(maxlen=int(self.p.get("window", 3)))
        self.buf: deque[np.ndarray] = deque(maxlen=int(self.p.get("tstd_window", 8)))
        self.noise = None  # EMA năng lượng nền khi quạt đang TẮT

    def measure(self, frame, scene, ts):
        g = crop(scene.gray, self.roi)
        # Kích thước cố định theo ROI cấu hình: ROI dịch lệch 1 px do làm tròn không làm reset bộ đệm.
        H, W = scene.gray.shape[:2]
        rw, rh = (self.cfg.roi[2] - self.cfg.roi[0]) * W, (self.cfg.roi[3] - self.cfg.roi[1]) * H
        side = int(self.p.get("resize", 128))
        s = side / max(rw, rh)
        size = (max(8, int(rw * s)), max(8, int(rh * s)))
        g = cv2.GaussianBlur(cv2.resize(g, size, interpolation=cv2.INTER_AREA), (3, 3), 0)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        pm = self._person_mask(scene, g.shape[:2])
        valid = ~pm if pm is not None else np.ones(g.shape[:2], bool)

        if scene.dark:
            self.prev = None
            self.buf.clear()
            return UNKNOWN, 0.0, {"dark": True}
        if self.camera_moved or self.prev is None or self.prev.shape != g.shape:
            self.prev = g
            self.buf.clear()
            self.buf.append(g.astype(np.float32))
            return UNKNOWN, 0.0, {"sharp": round(sharp, 1), "camera_moved": self.camera_moved}

        flow = cv2.calcOpticalFlowFarneback(self.prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = g
        mag = np.linalg.norm(flow, axis=2)
        energy = float(mag[valid].mean()) if valid.any() else 0.0
        self.energies.append(energy)
        e = float(np.mean(self.energies))

        self.buf.append(g.astype(np.float32))
        tfrac = 0.0
        if len(self.buf) >= int(self.p.get("tstd_min", 4)) and valid.any():
            tstd = np.stack(self.buf).std(axis=0)
            tfrac = float((tstd[valid] > self.p.get("tstd_pix", 4.0)).mean())

        base = self.p.get("flow_threshold", 0.6)
        thr = base if self.noise is None else max(base, self.p.get("noise_k", 4.0) * self.noise)
        sharp_off = self.p.get("sharp_off")
        blurred = bool(sharp_off is not None and sharp < self.p.get("blur_ratio", 0.5) * sharp_off)
        on = e >= thr or blurred or tfrac >= self.p.get("tstd_frac_on", 0.15)
        if self.state == OFF and not on:
            self.noise = e if self.noise is None else 0.95 * self.noise + 0.05 * e
        feats = {"flow": round(e, 3), "thr": round(thr, 3), "tstd_frac": round(tfrac, 3),
                 "sharp": round(sharp, 1), "blurred": blurred}
        return (ON if on else OFF), max(e, tfrac), feats


class FanAutoDetector(DeviceDetector):
    """Tự động tìm quạt đang quay trên toàn khung, không cần ROI hay kéo chọn vùng.

    Cánh quạt quay tạo một vùng cố định có độ sáng dao động liên tục theo thời gian (kể cả
    cánh trắng trên trần trắng). Thuật toán:
      1. Ảnh xám thu nhỏ; loại điểm ảnh thuộc người (mặt nạ phân đoạn, nới rộng) và vùng đèn
         sáng/quầng sáng (tránh nhấp nháy đèn huỳnh quang).
      2. Kiểm tra camera đứng yên (tương quan pha giữa hai khung liên tiếp); đang lia → xoá bộ đệm.
      3. Độ lệch chuẩn theo thời gian trên cửa sổ N khung → mặt nạ "dao động".
      4. Bản đồ nhiệt bền vững (trung bình trượt mũ): vùng dao động *liên tục* nhiều giây mới được
         giữ → loại cử động ngắt quãng của người.
      5. Thành phần liên thông đủ lớn, đủ đặc → mỗi thành phần là một quạt đang quay.
    Quạt TẮT không tạo tín hiệu → trạng thái OFF (đúng nhu cầu: chỉ cần biết quạt nào đang chạy).
    """

    occlusion_max = 1.1

    def __init__(self, cfg):
        super().__init__(cfg)
        self.width = int(self.p.get("width", 160))
        self.buf: deque[np.ndarray] = deque(maxlen=int(self.p.get("window", 8)))
        self.prev: np.ndarray | None = None
        self.heat: np.ndarray | None = None
        self.blobs: list[list[float]] = []
        self.dynamic_power: float | None = None
        self.win = None

    def measure(self, frame, scene, ts):
        H, W = scene.gray.shape[:2]
        s = self.width / W
        size = (self.width, int(H * s))
        g = cv2.GaussianBlur(cv2.resize(scene.gray, size, interpolation=cv2.INTER_AREA), (3, 3), 0).astype(np.float32)

        # Vùng loại trừ: người (nới rộng) + đèn sáng và quầng sáng.
        excl = scene.hsv[..., 2] >= self.p.get("bright_v", 225)
        if scene.person_mask is not None:
            excl |= scene.person_mask
        excl = cv2.dilate(excl.astype(np.uint8), np.ones((21, 21), np.uint8))
        excl = cv2.resize(excl, size, interpolation=cv2.INTER_NEAREST).astype(bool)

        if scene.dark:
            self.buf.clear()
            self.heat = None
            self.blobs = []
            return UNKNOWN, 0.0, {"dark": True}

        # Camera có đang di chuyển không (so với khung trước)?
        moving = False
        if self.prev is not None:
            if self.win is None or self.win.shape != g.shape:
                self.win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            fill = float(np.median(g[~excl])) if (~excl).any() else float(g.mean())
            a, b = self.prev.copy(), g.copy()
            a[excl], b[excl] = fill, fill
            (dx, dy), _ = cv2.phaseCorrelate(a, b, self.win)
            moving = np.hypot(dx, dy) / s > self.p.get("moving_px", 3.0)
        self.prev = g
        if moving:
            self.buf.clear()
            self.heat = None
            return UNKNOWN, 0.0, {"camera_moved": True, "count": len(self.blobs)}

        self.buf.append(g)
        if len(self.buf) < int(self.p.get("min_frames", 5)):
            return UNKNOWN, 0.0, {"warming_up": len(self.buf)}

        # Trừ độ sáng trung bình (vùng hợp lệ) của từng khung → loại thay đổi phơi sáng tự động.
        stack = np.stack(self.buf)
        valid = ~excl
        if valid.any():
            stack = stack - stack[:, valid].mean(axis=1)[:, None, None]
        tstd = stack.std(axis=0)
        act = ((tstd > self.p.get("tstd_pix", 4.0)) & valid).astype(np.float32)
        active = float(act[valid].mean()) if valid.any() else 0.0
        if active > self.p.get("global_max", 0.4):
            # Cả khung dao động (camera trôi chậm, rung, đổi phơi sáng mạnh) → không đo được.
            self.buf.clear()
            self.heat = None
            return UNKNOWN, 0.0, {"global_change": round(active, 2), "count": len(self.blobs)}
        a = self.p.get("heat_alpha", 0.3)
        self.heat = act if self.heat is None else (1 - a) * self.heat + a * act
        m = (self.heat >= self.p.get("heat_on", 0.5)).astype(np.uint8)

        if scene.device_boxes is not None and "fan" in scene.device_boxes:
            # Có YOLO: chỉ xét dao động bên trong box quạt → người/cửa cử động ngoài box không gây báo sai.
            mh, mw = m.shape[:2]
            blobs = []
            for bx in scene.device_boxes["fan"]:
                x1, y1, x2, y2 = _box_px(bx, mw, mh)
                ok = valid[y1:y2, x1:x2]
                if ok.sum() < 10:
                    continue
                if m[y1:y2, x1:x2][ok].mean() >= self.p.get("box_active_frac", 0.08):
                    blobs.append(bx)
            self.blobs = blobs
            self.dynamic_power = len(blobs) * float(self.p.get("per_fan_w", 60))
            return (ON if blobs else OFF), float(len(blobs)), \
                {"count": len(blobs), "fans_seen": len(scene.device_boxes["fan"]), "active_frac": round(active, 3)}

        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        area = m.shape[0] * m.shape[1]
        blobs = []
        for i in range(1, n):
            x, y, bw, bh, ar = stats[i]
            if not (self.p.get("min_area", 0.004) * area <= ar <= self.p.get("max_area", 0.3) * area):
                continue
            if ar / float(bw * bh) < self.p.get("fill_min", 0.25):
                continue
            # Vùng quét cánh quạt khá cân đối; dải dài-mảnh thường là mép người/cửa cử động.
            if max(bw, bh) / float(max(1, min(bw, bh))) > self.p.get("max_aspect", 3.0):
                continue
            blobs.append([round(x / m.shape[1], 4), round(y / m.shape[0], 4),
                          round((x + bw) / m.shape[1], 4), round((y + bh) / m.shape[0], 4)])
        self.blobs = blobs
        self.dynamic_power = len(blobs) * float(self.p.get("per_fan_w", 60))
        return (ON if blobs else OFF), float(len(blobs)), {"count": len(blobs), "active_frac": round(active, 3)}


class ScreenDetector(DeviceDetector):
    """Máy chiếu/màn chiếu, màn hình/TV: độ sáng tương đối và phương sai kết cấu."""

    def measure(self, frame, scene, ts):
        v = self._v(scene, self.roi).astype(np.float32)
        v_roi = float(v.mean())
        std = float(v.std())
        ratio = v_roi / (self._ref_v(scene) + 1.0)
        on = (ratio >= self.p.get("on_ratio", 1.3) and v_roi >= self.p.get("min_v", 60)) or \
             (std >= self.p.get("std_on", 40.0) and v_roi >= self.p.get("min_v", 60))
        return (ON if on else OFF), ratio, {"ratio": round(ratio, 3), "std": round(std, 1), "v_roi": round(v_roi, 1)}


class ACDetector(DeviceDetector):
    """Điều hoà: đèn báo trên dàn lạnh — tỉ lệ điểm ảnh sáng và có màu trong ROI nhỏ."""

    def measure(self, frame, scene, ts):
        hsv = crop(scene.hsv, self.roi)
        led = (hsv[..., 2] >= self.p.get("led_v", 200)) & (hsv[..., 1] >= self.p.get("led_s", 80))
        frac = float(led.mean())
        on = frac >= self.p.get("led_frac", 0.01)
        return (ON if on else OFF), frac, {"led_frac": round(frac, 4)}


DETECTORS = {
    "light": LightDetector,
    "lights_auto": LightGroupDetector,
    "fan": FanDetector,
    "fan_auto": FanAutoDetector,
    "screen": ScreenDetector,
    "projector": ScreenDetector,
    "monitor": ScreenDetector,
    "tv": ScreenDetector,
    "ac": ACDetector,
}


def build_detector(cfg: DeviceConfig) -> DeviceDetector:
    try:
        return DETECTORS[cfg.type](cfg)
    except KeyError:
        raise ValueError(f"Loại thiết bị không hỗ trợ: {cfg.type} (hỗ trợ: {sorted(DETECTORS)})")


class ROIClassifier:
    """Bộ phân lớp nhị phân MobileNetV3 trên ROI (lớp 0 = off, 1 = on).

    Hỗ trợ .onnx (onnxruntime — khuyến nghị trên Raspberry Pi) và .pt (TorchScript).
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, path: str, size: int | list[int] = 224):
        # size: cạnh ảnh vuông, hoặc [rộng, cao] (bộ phân lớp toàn khung: [384, 224])
        self.size = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        self.path = path
        if path.endswith(".onnx"):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input = self.sess.get_inputs()[0].name
            self.torch_model = None
        else:
            import torch
            self.torch_model = torch.jit.load(path, map_location="cpu").eval()
            self.sess = None

    def preprocess(self, roi_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(roi_bgr, self.size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return img.transpose(2, 0, 1)[None].astype(np.float32)

    def prob_on(self, roi_bgr: np.ndarray) -> float:
        x = self.preprocess(roi_bgr)
        if self.sess is not None:
            logits = self.sess.run(None, {self.input: x})[0][0]
        else:
            import torch
            with torch.no_grad():
                logits = self.torch_model(torch.from_numpy(x))[0].numpy()
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())
