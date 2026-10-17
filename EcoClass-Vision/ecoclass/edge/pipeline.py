"""Quy trình xử lý tại biên của EcoClass-Vision (Hình 1, tầng xử lý tại biên).

camera → lấy mẫu khung → [YOLO11 + ByteTrack → N(t)] + [ROI → trạng thái thiết bị]
       → máy trạng thái có trễ → (làm mờ người) → sự kiện JSON → MQTT/HTTP
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import RoomConfig
from ..fsm import ON, OFF, RoomState, TemporalFSM
from ..timetable import Timetable
from .capture import FrameSource
from .devices import build_detector, make_scene
from .geometry import px_to_roi, roi_to_px
from .motion import CameraMotion
from .person import PersonCounter
from .privacy import blur_boxes, blur_mask, encode_jpeg
from .publisher import Publisher

COLORS = {ON: (0, 0, 255), OFF: (0, 200, 0)}
STATE_COLORS = {
    "OCCUPIED": (235, 150, 60), "EMPTY_PENDING": (0, 200, 255), "ALERT": (0, 0, 255),
    "RESOLVED": (0, 200, 0), "EMPTY_SAFE": (0, 160, 0),
}


def _json_default(o):
    """Kiểu numpy (np.bool_, np.float32...) → kiểu Python để ghi JSON."""
    return o.item() if hasattr(o, "item") else str(o)


class EdgePipeline:
    def __init__(
        self,
        cfg: RoomConfig,
        publisher: Publisher | None = None,
        use_tracker: bool = True,
        temporal: bool = True,
        frame_log: str | None = None,
        sensor=None,
    ):
        self.cfg = cfg
        self.publisher = publisher
        if sensor is None and cfg.sensor.enabled:
            from .esp32 import ESP32Sensor
            sensor = ESP32Sensor(cfg.sensor.port, cfg.sensor.baud, cfg.sensor.stale_seconds).start()
        self.sensor = sensor          # ESP32Sensor hoặc vật giả có .latest() (kiểm thử)
        self.counter = PersonCounter(cfg.person, use_tracker=use_tracker, model_path=cfg.resolve(cfg.person.model))
        for d in cfg.devices:
            d.classifier = cfg.resolve(d.classifier)
        self.detectors = {d.id: build_detector(d) for d in cfg.devices}
        self.motion = CameraMotion() if any(d.params.get("track") for d in cfg.devices) else None
        self.locator = None
        model = cfg.resolve(cfg.device_model)
        if model:
            if Path(model).exists():
                from .device_yolo import DeviceLocator
                self.locator = DeviceLocator(model, conf=cfg.device_conf, device=cfg.person.device)
                print(f"[edge] bộ phát hiện thiết bị: {model} {self.locator.names}")
            else:
                print(f"[edge] chưa có {model} → dùng luật dò toàn khung")
        self.timetable = Timetable.load(cfg.resolve(cfg.timetable))
        fsm_cfg = cfg.fsm
        if not temporal:
            # Ablation: bỏ bộ suy luận theo thời gian → cảnh báo ngay khi N=0 và có thiết bị bật.
            from dataclasses import replace
            fsm_cfg = replace(fsm_cfg, empty_seconds_to_pending=0.0, te_seconds=0.0,
                              te_no_class_seconds=0.0, off_confirm_seconds=0.0)
        self.fsm = TemporalFSM(cfg.room_id, fsm_cfg, {d.id: d.power_w for d in cfg.devices},
                               te_fn=self._te, occupied_confirm_updates=2 if temporal else 1)
        if frame_log:
            Path(frame_log).parent.mkdir(parents=True, exist_ok=True)
        self.frame_log = open(frame_log, "w", encoding="utf-8", buffering=1) if frame_log else None
        self.events: list[dict] = []
        self._last_hb = 0.0
        self._last_frame = 0.0
        self.fps: float | None = None   # tốc độ xử lý thực tế (run() cập nhật) → gửi kèm heartbeat

    def _te(self, ts: float) -> float:
        if not self.timetable.slots:
            return self.fsm.cfg.te_seconds
        when = datetime.fromtimestamp(ts)
        in_class = self.timetable.in_session(self.cfg.room_id, when)
        return self.fsm.cfg.te_seconds if in_class else self.fsm.cfg.te_no_class_seconds

    def process(self, frame: np.ndarray, ts: float) -> dict:
        occupied = self.fsm.state == RoomState.OCCUPIED
        pr = self.counter.update(frame, ts, occupied=occupied)
        dev_boxes = self.locator(frame) if self.locator is not None else None
        sensors = self.sensor.latest() if self.sensor is not None else None
        scene = make_scene(frame, pr.boxes, pr.mask, motion=self.motion, device_boxes=dev_boxes, sensors=sensors)
        readings = {k: det.update(frame, scene, ts) for k, det in self.detectors.items()}
        states = {k: r.state for k, r in readings.items()}
        for k, det in self.detectors.items():
            # Nhóm đèn tự dò: công suất = số đèn đang sáng thấy được × công suất mỗi đèn.
            if getattr(det, "dynamic_power", None) and readings[k].raw == ON:
                self.fsm.power_w[k] = det.dynamic_power

        events = self.fsm.update(ts, pr.n, states)
        for ev in events:
            d = ev.to_dict()
            d["iso_time"] = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
            d["in_class"] = bool(self.timetable.slots) and self.timetable.in_session(self.cfg.room_id, datetime.fromtimestamp(ts))
            d["power_w"] = {k: self.detectors[k].cfg.power_w for k in d["devices_on"]}
            d["tariff_vnd_per_kwh"] = self.cfg.tariff_vnd_per_kwh
            if sensors:
                d["temperature"], d["humidity"] = sensors["temperature"], sensors["humidity"]
            jpeg = None
            if ev.type == "alert":
                jpeg = encode_jpeg(self.render_evidence(frame, pr.boxes, states), self.cfg.evidence_max_width)
            self.events.append(d)
            if self.publisher:
                self.publisher.publish_event(d, jpeg)

        # Hiển thị: số người thấy trong khung hiện tại. pr.n (dùng cho máy trạng thái) còn tính cả vết vừa
        # mất dấu trong hold_seconds để không báo "vắng" giả → khi ID đổi liên tục, pr.n đếm dư.
        snap = self.fsm.snapshot(ts, len(pr.boxes) if pr.boxes else pr.n, states)
        snap["dark"] = scene.dark
        if self.fps is not None:
            snap["fps"] = round(self.fps, 1)
        if sensors is not None:
            snap["sensors"] = sensors
        if self.publisher and ts - self._last_hb >= self.cfg.publisher.heartbeat_seconds:
            self._last_hb = ts
            self.publisher.publish_state(snap)

        if self.frame_log:
            self.frame_log.write(json.dumps({
                "ts": round(ts, 3), "n_raw": pr.n_raw, "n": pr.n, "state": snap["state"], "devices": states,
                "raw": {k: r.raw for k, r in readings.items()},
                "features": {k: r.features for k, r in readings.items()},
            }, ensure_ascii=False, default=_json_default) + "\n")

        out = {"person": pr, "readings": readings, "states": states, "events": events, "snapshot": snap}
        pc = self.cfg.publisher
        # Dung sai 20%: khung camera đến lệch nhịp vài ms, so sánh chặt sẽ bỏ gần nửa số khung.
        if self.publisher and pc.frame_fps > 0 and ts - self._last_frame >= 0.8 / pc.frame_fps:
            self._last_frame = ts
            self.publisher.publish_frame(encode_jpeg(self.render_stream(frame, pr), pc.frame_width, quality=75))
        return out

    def render_stream(self, frame, pr) -> np.ndarray:
        """Ảnh cho màn hình camera: người đã làm mờ + khung người mảnh, không chữ
        (trạng thái/nhiệt độ do trang web hiển thị)."""
        img = blur_mask(frame, pr.mask) if pr.mask is not None else blur_boxes(frame, pr.boxes)
        th = max(1, img.shape[1] // 400)
        for b in pr.boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 200, 0), th, cv2.LINE_AA)
        return img

    def _device_boxes(self, det) -> list[list[float]]:
        """Các vùng cần vẽ cho thiết bị: ROI hiện tại (đã bám) hoặc các đèn tự dò."""
        if det.cfg.type == "ac_temp":      # đo bằng cảm biến, không có vùng trên ảnh
            return []
        blobs = getattr(det, "blobs", None)
        return blobs if blobs is not None else [det.roi]

    def render_evidence(self, frame, person_boxes, states) -> np.ndarray:
        img = blur_boxes(frame, person_boxes)
        h, w = img.shape[:2]
        for k, det in self.detectors.items():
            if states.get(k) == ON:
                for roi in self._device_boxes(det):
                    x1, y1, x2, y2 = roi_to_px(roi, w, h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, f"{k} ON", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        stamp = f"{self.cfg.room_id}  {datetime.fromtimestamp(self.fsm.last_ts or time.time()):%Y-%m-%d %H:%M:%S}"
        cv2.putText(img, stamp, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def render_debug(self, frame, out: dict, scale: float = 1.0) -> np.ndarray:
        """Ảnh gỡ lỗi (người luôn được làm mờ). ``scale`` > 1: phóng to + làm nét trước khi vẽ
        để chữ/khung sắc nét khi webcam chỉ cho độ phân giải thấp."""
        pr = out["person"]
        # Có mặt nạ phân đoạn → chỉ làm mờ đúng hình người, phần nền giữ nét.
        img = blur_mask(frame, pr.mask) if pr.mask is not None else blur_boxes(frame, pr.boxes)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 2), -0.5, 0)   # unsharp mask
            from .person import PersonResult
            pr = PersonResult(pr.n_raw, pr.n, [tuple(v * scale for v in b) for b in pr.boxes], pr.confs, pr.ids)
        h, w = img.shape[:2]
        sc = max(0.5, w / 1280)                       # cỡ chữ/nét theo độ phân giải
        th = max(1, int(round(2 * sc)))
        font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA

        def text(t, org, color, scale=0.6):
            cv2.putText(img, t, org, font, scale * sc * 1.4, (0, 0, 0), th + 2, aa)   # viền đen cho dễ đọc
            cv2.putText(img, t, org, font, scale * sc * 1.4, color, th, aa)

        for b, i in zip(pr.boxes, pr.ids):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 200, 0), th, aa)
            text(f"id{i}", (int(b[0]) + 4, int(b[1]) + int(22 * sc)), (255, 200, 0), 0.5)
        free_row = 0                                  # nhãn của thiết bị không có khung → xếp dòng ở góc trái
        for k, det in self.detectors.items():
            r = out["readings"][k]
            c = COLORS.get(r.state, (160, 160, 160))
            label = f"{k}:{r.state}" + ("" if r.raw != "UNKNOWN" else "?")
            if "count" in r.features:
                label += f" ({r.features['count']} {'quat' if det.cfg.type == 'fan_auto' else 'den'})"
            if "temperature" in r.features:
                label += f" ({r.features['temperature']:.1f}C)"
            boxes = self._device_boxes(det)
            for roi in boxes:
                x1, y1, x2, y2 = roi_to_px(roi, w, h)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, th, aa)
            if boxes:
                x1, y1 = roi_to_px(boxes[0], w, h)[:2]
                text(label, (x1 + 2, max(int(70 * sc), y1 - 6)), c)
            else:
                free_row += 1
                text(label, (10, int((44 + 30 * free_row) * sc)), c)
        s = out["snapshot"]
        bar = int(44 * sc)
        cv2.rectangle(img, (0, 0), (w, bar), (30, 30, 30), -1)
        txt = f"{s['state']}  N={s['people']}  empty={s['empty_seconds']:.0f}s/Te={s['te_seconds']:.0f}s  waste={s['waste_wh_current']:.1f}Wh"
        cv2.putText(img, txt, (10, int(31 * sc)), font, 0.9 * sc, STATE_COLORS.get(s["state"], (255, 255, 255)), th, aa)
        if "fps" in out:
            ft = f"{out['fps']:.1f} FPS"
            (tw, _), _ = cv2.getTextSize(ft, font, 0.8 * sc, th)
            cv2.putText(img, ft, (w - tw - 10, int(31 * sc)), font, 0.8 * sc, (200, 200, 200), th, aa)
        return img

    def close(self) -> None:
        if self.frame_log:
            self.frame_log.close()


def run(
    cfg: RoomConfig,
    publisher: Publisher | None,
    show: bool = False,
    save_video: str | None = None,
    use_tracker: bool = True,
    temporal: bool = True,
    frame_log: str | None = None,
    t0: float | None = None,
    realtime: bool = False,
    max_frames: int | None = None,
    select_roi: list[str] | None = None,
    loop: bool = False,
) -> list[dict]:
    source = FrameSource(cfg.resolve(cfg.source), fps=cfg.fps, t0=t0, realtime=realtime,
                         width=cfg.capture_width, height=cfg.capture_height, flip=cfg.flip, loop=loop,
                         rotate=cfg.rotate, max_side=cfg.max_side)
    title = f"EcoClass-Vision {cfg.room_id}"
    if show:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # cửa sổ kéo giãn được
    frames = iter(source)
    first = next(frames)
    if select_roi:
        _select_rois(cfg, first.image, select_roi)
    pipe = EdgePipeline(cfg, publisher, use_tracker=use_tracker, temporal=temporal, frame_log=frame_log)
    writer = None
    n, t_start = 0, time.time()
    t_prev, fps_ema = time.time(), None
    try:
        for fr in itertools.chain([first], frames):
            out = pipe.process(fr.image, fr.ts)
            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
            t_prev = now
            out["fps"] = pipe.fps = fps_ema
            for ev in out["events"]:
                extra = f" waste={ev.waste_wh:.2f}Wh reason={ev.reason}" if ev.type != "state_change" else f" ({ev.reason})"
                print(f"[{datetime.fromtimestamp(ev.ts):%H:%M:%S}] {ev.type:13s} {ev.state_from} -> {ev.state_to}"
                      f" N={ev.people} on={ev.devices_on}{extra}")
            if show or save_video:
                dbg = pipe.render_debug(fr.image, out, scale=cfg.display_scale)
                if save_video:
                    if writer is None:
                        Path(save_video).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps,
                                                 (dbg.shape[1], dbg.shape[0]))
                    writer.write(dbg)
                if show:
                    cv2.imshow(title, dbg)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            n += 1
            if max_frames and n >= max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        el = time.time() - t_start
        print(f"[edge] {n} khung trong {el:.1f}s → {n / max(el, 1e-6):.2f} FPS xử lý")
        pipe.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if publisher:
            publisher.flush()
    return pipe.events


def _select_rois(cfg: RoomConfig, image: np.ndarray, ids: list[str]) -> None:
    """Kéo chuột chọn ROI cho các thiết bị trên khung đầu tiên (thay cho Auto-ROI khi YOLO-World không thấy)."""
    h, w = image.shape[:2]
    by_id = {d.id: d for d in cfg.devices}
    for i in ids:
        if i not in by_id:
            raise ValueError(f"Không có thiết bị {i} trong cấu hình")
        title = f"Keo chuot chon vung {i} roi nhan ENTER (ESC = giu nguyen)"
        x, y, bw, bh = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if bw > 4 and bh > 4:
            by_id[i].roi = px_to_roi((x, y, x + bw, y + bh), w, h)
            print(f"[roi] {i}: {by_id[i].roi}")
