"""Nhận bản tin từ broker MQTT (Mosquitto) và chuyển vào cùng hàm xử lý với HTTP ingest."""
from __future__ import annotations

import json
from typing import Callable


def start_mqtt_bridge(host: str, port: int, prefix: str,
                      on_state: Callable[[dict], None], on_event: Callable[[dict], None],
                      password: str | None = None):
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ecoclass-server")
    if password:
        client.username_pw_set("server", password)

    def on_connect(c, userdata, flags, rc, props=None):
        print(f"[mqtt] kết nối {host}:{port} rc={rc}")
        c.subscribe(f"{prefix}/+/state", qos=1)
        c.subscribe(f"{prefix}/+/event", qos=1)

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            (on_state if msg.topic.endswith("/state") else on_event)(payload)
        except Exception as e:
            print(f"[mqtt] lỗi xử lý {msg.topic}: {e}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(host, port, keepalive=30)
    client.loop_start()
    return client
