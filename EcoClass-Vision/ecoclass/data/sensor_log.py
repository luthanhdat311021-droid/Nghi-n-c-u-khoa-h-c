"""Ghi log cảm biến điện năng PZEM-004T (qua ESP32 → MQTT) ra CSV.

Chủ đề: ecoclass/{room_id}/power/{device_id}
Payload: {"ts": 1759300000.123, "voltage": 228.1, "current": 0.31, "power": 70.2, "energy": 12.5}

    python -m ecoclass.data.sensor_log --host 127.0.0.1 --out datasets/sensor_A1-101.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def main() -> None:
    import paho.mqtt.client as mqtt

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--prefix", default="ecoclass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["ts", "room_id", "device_id", "voltage", "current", "power", "energy"])

    def on_connect(c, u, flags, rc, props=None):
        c.subscribe(f"{a.prefix}/+/power/+", qos=1)
        print("[sensor_log] đã kết nối, đang ghi", path)

    def on_message(c, u, msg):
        _, room, _, dev = msg.topic.split("/")[-4:]
        p = json.loads(msg.payload)
        ts = p.get("ts") or time.time()  # ESP32 đồng bộ NTP; nếu thiếu dùng giờ máy nhận
        w.writerow([f"{ts:.3f}", room, dev, p.get("voltage"), p.get("current"), p.get("power"), p.get("energy")])
        f.flush()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ecoclass-sensor-log")
    c.on_connect, c.on_message = on_connect, on_message
    c.connect(a.host, a.port, 30)
    try:
        c.loop_forever()
    finally:
        f.close()


if __name__ == "__main__":
    main()
