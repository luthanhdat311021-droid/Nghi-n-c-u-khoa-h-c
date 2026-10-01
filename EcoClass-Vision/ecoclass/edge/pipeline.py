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
    ):
        self.cfg = cfg
        self.publisher = publisher
        self.counter = PersonCounter(cfg.person, use_tracker=use_tracker, model_path=cfg.resolve(cfg.person.model))
        for d in cfg.devices:
            d.classifier = cfg.resolve(d.classifier)
        self.detectors = {d.id: build_detector(d) for d in cfg.devices}
        self.motion = CameraMotion() if any(d.params.get("track") for d in cfg.devices) else None
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

    def _te(self, ts: float) -> float:
        if not self.timetable.slots:
            return self.fsm.cfg.te_seconds
        when = datetime.fromtimestamp(ts)
        in_class = self.timetable.in_session(self.cfg.room_id, when)
        return self.fsm.cfg.te_seconds if in_class else self.fsm.cfg.te_no_class_seconds

    def process(self, frame: np.ndarray, ts: float) -> dict:
        occupied = self.fsm.state == RoomState.OCCUPIED
        pr = self.counter.update(frame, ts, occupied=occupied)
        scene = make_scene(frame, pr.boxes, pr.mask, motion=self.motion)
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
            jpeg = None
            if ev.type == "alert":
                jpeg = encode_jpeg(self.render_evidence(frame, pr.boxes, states), self.cfg.evidence_max_width)
            self.events.append(d)
            if self.publisher:
                self.publisher.publish_event(d, jpeg)

        snap = self.fsm.snapshot(ts, pr.n, states)
        snap["dark"] = scene.dark
        if self.publisher and ts - self._last_hb >= self.cfg.publisher.heartbeat_seconds:
            self._last_hb = ts
            self.publisher.publish_state(snap)

        if self.frame_log:
            self.frame_log.write(json.dumps({
                "ts": round(ts, 3), "n_raw": pr.n_raw, "n": pr.n, "state": snap["state"], "devices": states,
                "raw": {k: r.raw for k, r in readings.items()},
                "features": {k: r.features for k, r in readings.items()},
            }, ensure_ascii=False, default=_json_default) + "\n")

        return {"person": pr, "readings": readings, "states": states, "events": events, "snapshot": snap}

    def _device_boxes(self, det) -> list[list[float]]:
        """Các vùng cần vẽ cho thiết bị: ROI hiện tại (đã bám) hoặc các đèn tự dò."""
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

    def render_debug(self, frame, out: dict) -> np.ndarray:
        """Ảnh gỡ lỗi (người luôn được làm mờ)."""
        pr = out["person"]
        # Có mặt nạ phân đoạn → chỉ làm mờ đúng hình người, phần nền giữ nét.
        img = blur_mask(frame, pr.mask) if pr.mask is not None else blur_boxes(frame, pr.boxes)
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
        for k, det in self.detectors.items():
            r = out["readings"][k]
            c = COLORS.get(r.state, (160, 160, 160))
            label = f"{k}:{r.state}" + ("" if r.raw != "UNKNOWN" else "?")
            if "count" in r.features:
                label += f" ({r.features['count']} {'quat' if det.cfg.type == 'fan_auto' else 'den'})"
            boxes = self._device_boxes(det)
            for roi in boxes:
                x1, y1, x2, y2 = roi_to_px(roi, w, h)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, th, aa)
            x1, y1 = roi_to_px(boxes[0], w, h)[:2] if boxes else (8, int(80 * sc))
            text(label, (x1 + 2, max(int(70 * sc), y1 - 6)), c)
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
) -> list[dict]:
    source = FrameSource(cfg.resolve(cfg.source), fps=cfg.fps, t0=t0, realtime=realtime,
                         width=cfg.capture_width, height=cfg.capture_height)
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
            out["fps"] = fps_ema
            for ev in out["events"]:
                extra = f" waste={ev.waste_wh:.2f}Wh reason={ev.reason}" if ev.type != "state_change" else f" ({ev.reason})"
                print(f"[{datetime.fromtimestamp(ev.ts):%H:%M:%S}] {ev.type:13s} {ev.state_from} -> {ev.state_to}"
                      f" N={ev.people} on={ev.devices_on}{extra}")
            if show or save_video:
                dbg = pipe.render_debug(fr.image, out)
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
