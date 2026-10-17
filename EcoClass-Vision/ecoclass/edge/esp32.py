"""Đọc nhiệt độ/độ ẩm từ ESP32 + DHT11 qua cổng USB (serial), chạy nền liên tục.

Firmware (sketch_oct16a.ino) in mỗi giây một dòng dạng:
    STA_STATUS=3 STA_IP=192.168.221.101 AP_CLIENTS=0 Temperature_C=27.5 Humidity_percent=56.4

- ``port="auto"``: tự tìm cổng của mạch USB-UART (CH340 / CP210x / FTDI). Cắm sang cổng USB khác
  (COM4 → COM5) vẫn chạy được.
- Rút cáp / mạch khởi động lại: tự kết nối lại sau vài giây, không làm dừng quy trình camera.
- Mở cổng với DTR/RTS = False để không reset ESP32.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

LINE_RE = re.compile(r"Temperature_C=(-?[\d.]+)\s+Humidity_percent=([\d.]+)")
IP_RE = re.compile(r"STA_IP=(\S+)")
USB_UART_VIDS = {0x1A86, 0x10C4, 0x0403, 0x303A}   # CH340, CP210x, FTDI, Espressif USB


@dataclass
class SensorReading:
    temperature: float | None = None
    humidity: float | None = None
    ts: float | None = None          # thời điểm đọc được (epoch)
    esp_ip: str | None = None
    error: bool = False              # dòng cuối là DHT_READ_ERROR


def parse_line(line: str, prev: SensorReading, ts: float) -> SensorReading:
    """Cập nhật kết quả đo từ một dòng log của firmware (hàm thuần, dễ kiểm thử)."""
    ip = IP_RE.search(line)
    esp_ip = ip.group(1) if ip else prev.esp_ip
    m = LINE_RE.search(line)
    if m:
        return SensorReading(float(m.group(1)), float(m.group(2)), ts, esp_ip, False)
    if "DHT_READ_ERROR" in line:
        return SensorReading(prev.temperature, prev.humidity, prev.ts, esp_ip, True)
    return SensorReading(prev.temperature, prev.humidity, prev.ts, esp_ip, prev.error)


def find_port() -> str | None:
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in USB_UART_VIDS:
            return p.device
    return None


class ESP32Sensor:
    def __init__(self, port: str = "auto", baud: int = 115200, stale_seconds: float = 10.0):
        self.port_cfg = port
        self.baud = baud
        self.stale_seconds = stale_seconds
        self.port: str | None = None
        self.connected = False
        self._last_msg: str | None = None
        self._reading = SensorReading()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True, name="esp32")

    def start(self) -> "ESP32Sensor":
        self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def latest(self, now: float | None = None) -> dict:
        """Kết quả mới nhất; ``ok`` = False khi mất kết nối hoặc số liệu cũ hơn ``stale_seconds``."""
        now = time.time() if now is None else now
        with self._lock:
            r = self._reading
        fresh = r.ts is not None and now - r.ts <= self.stale_seconds
        return {
            "temperature": r.temperature if fresh else None,
            "humidity": r.humidity if fresh else None,
            "sensor_ok": bool(fresh and self.connected),
            "sensor_port": self.port,
            "esp_ip": r.esp_ip if r.esp_ip and r.esp_ip != "0.0.0.0" else None,
            "sensor_age": round(now - r.ts, 1) if r.ts is not None else None,
        }

    def _run(self) -> None:
        import serial

        while not self._stop.is_set():
            port = find_port() if self.port_cfg == "auto" else self.port_cfg
            if not port:
                self._set_disconnected("không thấy mạch ESP32 (cắm cáp USB?)")
                time.sleep(3)
                continue
            try:
                ser = serial.Serial()
                ser.port, ser.baudrate, ser.timeout = port, self.baud, 1
                ser.dtr = ser.rts = False                 # không reset ESP32 khi mở cổng
                ser.open()
            except Exception as e:
                self._set_disconnected(f"không mở được {port}: {e}")
                time.sleep(3)
                continue
            self.port, self.connected, self._last_msg = port, True, None
            print(f"[esp32] đã kết nối {port}")
            try:
                while not self._stop.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    with self._lock:
                        self._reading = parse_line(line, self._reading, time.time())
            except Exception as e:                        # rút cáp → SerialException
                self._set_disconnected(f"mất kết nối {port}: {e}")
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
            time.sleep(2)

    def _set_disconnected(self, msg: str) -> None:
        if msg != self._last_msg:                         # không in lặp lại mỗi lần thử
            print(f"[esp32] {msg}")
            self._last_msg = msg
        self.connected = False
