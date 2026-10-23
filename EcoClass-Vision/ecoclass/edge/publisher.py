"""Gửi sự kiện JSON từ thiết bị biên lên máy chủ: MQTT (mặc định khi triển khai),
HTTP (khi không có broker) hoặc stdout (gỡ lỗi).

Chủ đề MQTT:
    {prefix}/{room_id}/state   heartbeat (retain)
    {prefix}/{room_id}/event   sự kiện (alert, alert_closed, state_change), QoS 1
Ảnh bằng chứng (đã làm mờ người) được nhúng base64 trong trường ``evidence_jpeg_b64``.
Không bao giờ gửi video.

Lệnh từ máy chủ (vd tắt máy lạnh): chế độ HTTP nhận kèm trong phản hồi heartbeat (``commands``), mỗi lệnh
được chuyển cho ``on_command``.
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
from typing import Callable

from ..config import PublisherConfig


class Publisher:
    def __init__(self, cfg: PublisherConfig, room_id: str):
        self.cfg = cfg
        self.room_id = room_id
        self.q: queue.Queue = queue.Queue(maxsize=1000)
        self.on_command: Callable[[dict], None] | None = None
        self._client = None
        if cfg.mode == "mqtt":
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-{room_id}")
            if cfg.api_key:
                self._client.username_pw_set("edge", cfg.api_key)
            self._client.connect_async(cfg.mqtt_host, cfg.mqtt_port, keepalive=30)
            self._client.loop_start()
        elif cfg.mode not in ("http", "stdout"):
            raise ValueError(f"publisher.mode không hợp lệ: {cfg.mode}")
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()
        # Ảnh xem trực tiếp: chỉ giữ ảnh mới nhất, gửi ở luồng riêng → không chặn sự kiện/cảnh báo.
        self._frame: bytes | None = None
        self._frame_ready = threading.Event()
        if cfg.mode == "http":
            threading.Thread(target=self._frame_worker, daemon=True).start()

    def publish_frame(self, jpeg: bytes) -> None:
        self._frame = jpeg
        self._frame_ready.set()

    def _frame_worker(self) -> None:
        import requests

        session = requests.Session()
        headers = {"Content-Type": "image/jpeg", **({"X-API-Key": self.cfg.api_key} if self.cfg.api_key else {})}
        while True:
            self._frame_ready.wait()
            self._frame_ready.clear()
            jpeg = self._frame
            try:
                session.post(f"{self.cfg.http_url}/api/ingest/frame/{self.room_id}", data=jpeg, headers=headers,
                             timeout=5).raise_for_status()
            except Exception:
                time.sleep(2)          # máy chủ chưa sẵn sàng: bỏ ảnh này, thử ảnh sau

    def publish_state(self, snapshot: dict) -> None:
        self._put("state", snapshot)

    def publish_event(self, event: dict, evidence_jpeg: bytes | None = None) -> None:
        payload = dict(event)
        if evidence_jpeg:
            payload["evidence_jpeg_b64"] = base64.b64encode(evidence_jpeg).decode("ascii")
        self._put("event", payload)

    def _put(self, kind: str, payload: dict) -> None:
        try:
            self.q.put_nowait((kind, payload))
        except queue.Full:
            print("[publisher] hàng đợi đầy, bỏ bản tin", kind)

    def flush(self, timeout: float = 10.0) -> None:
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.2)

    def _worker(self) -> None:
        import requests

        session = requests.Session()
        headers = {"X-API-Key": self.cfg.api_key} if self.cfg.api_key else {}
        while True:
            kind, payload = self.q.get()
            for attempt in range(5):
                try:
                    self._send(kind, payload, session, headers)
                    break
                except Exception as e:  # mạng chập chờn: thử lại có giãn cách
                    print(f"[publisher] gửi {kind} lỗi ({e}); thử lại #{attempt + 1}")
                    time.sleep(min(2 ** attempt, 15))

    def _send(self, kind, payload, session, headers) -> None:
        if self.cfg.mode == "stdout":
            p = {k: v for k, v in payload.items() if k != "evidence_jpeg_b64"}
            if "evidence_jpeg_b64" in payload:
                p["evidence"] = f"<jpeg {len(payload['evidence_jpeg_b64']) * 3 // 4} bytes>"
            print(f"[{kind}]", json.dumps(p, ensure_ascii=False))
        elif self.cfg.mode == "http":
            r = session.post(f"{self.cfg.http_url}/api/ingest/{kind}", json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            if kind == "state" and self.on_command:
                try:
                    cmds = r.json().get("commands") or []
                except ValueError:
                    cmds = []
                for c in cmds:
                    try:
                        self.on_command(c)
                    except Exception as e:      # lỗi xử lý lệnh không được làm gửi lại heartbeat
                        print("[publisher] xử lý lệnh lỗi:", e)
        else:
            topic = f"{self.cfg.topic_prefix}/{self.room_id}/{kind}"
            info = self._client.publish(topic, json.dumps(payload, ensure_ascii=False),
                                        qos=1, retain=(kind == "state"))
            info.wait_for_publish(timeout=10)
            if not info.is_published():
                raise RuntimeError("MQTT chưa xác nhận")
