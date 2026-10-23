"""Chạy toàn bộ hệ thống quản lý phòng học và giữ nó chạy liên tục.

    python scripts/run_system.py                                  # webcam + ESP32, dashboard http://127.0.0.1:8000
    python scripts/run_system.py --source http://192.168.1.23:8080/video --show
    python scripts/run_system.py --config configs/room_system.yaml --port 8000

Camera Imou: thêm vào .env một dòng (mật khẩu = Safety Code dưới đáy camera, để nguyên dấu "!")
    IMOU_RTSP=rtsp://admin:<SAFETY_CODE>@192.168.1.12:554/cam/realmonitor?channel=1&subtype=1
→ chạy không cần --source/--config sẽ tự dùng camera này với configs/room_imou.yaml.

Khởi động 2 tiến trình con:
  1. máy chủ (FastAPI/uvicorn): CSDL, cảnh báo, dashboard
  2. thiết bị biên (scripts/run_edge.py): camera → người/đèn/quạt, ESP32 → nhiệt độ, máy trạng thái 2 phút
Tiến trình nào dừng (lỗi, mất camera, ...) sẽ được khởi động lại sau vài giây. Ctrl+C để tắt hết.
Nhật ký ghi thêm vào runs/logs/.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/room_system.yaml"
IMOU_CONFIG = "configs/room_imou.yaml"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def encode_password(url: str) -> str:
    """rtsp://admin:L2!x@ip/... → mã hoá mật khẩu (! → %21) để FFmpeg đọc đúng; mã sẵn rồi thì giữ nguyên."""
    m = re.match(r"^(\w+://[^:/@]+:)(.*)@([^@]+)$", url)
    return f"{m[1]}{quote(unquote(m[2]), safe='')}@{m[3]}" if m else url


def hide_password(url: str) -> str:
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


class Service:
    def __init__(self, name: str, cmd: list[str], log_dir: Path):
        self.name, self.cmd = name, cmd
        self.log_path = log_dir / f"{name}.log"
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.next_start = 0.0
        self.started_at = 0.0

    def start(self) -> None:
        self.started_at = time.time()
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        log(f"[{self.name}] khởi động (pid {self.proc.pid})")

    def _pump(self, proc: subprocess.Popen) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                print(f"[{self.name}] {line}", flush=True)
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
                f.flush()

    def check(self) -> None:
        now = time.time()
        if self.proc is None:
            if now >= self.next_start:
                self.start()
            return
        code = self.proc.poll()
        if code is None:
            return
        # Chạy ổn định > 60 s rồi mới dừng → khởi động lại ngay; dừng liên tục → giãn cách tới 60 s.
        self.restarts = 0 if now - self.started_at > 60 else self.restarts + 1
        delay = min(5 * 2 ** self.restarts, 60)
        log(f"[{self.name}] đã dừng (mã {code}); khởi động lại sau {delay} s")
        self.proc, self.next_start = None, now + delay

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help=f"mặc định {DEFAULT_CONFIG} ({IMOU_CONFIG} khi .env có IMOU_RTSP)")
    ap.add_argument("--source", help="ghi đè nguồn camera (vd http://<IP>:8080/video, rtsp://..., 0)")
    ap.add_argument("--flip", choices=["none", "horizontal", "vertical", "both"],
                    help="lật ảnh (mặc định: theo cấu hình với webcam, 'none' với camera IP/điện thoại)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270],
                    help="xoay ảnh theo chiều kim đồng hồ (điện thoại cầm dọc: thường 90)")
    ap.add_argument("--port", type=int, default=8000, help="cổng dashboard")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = máy khác trong mạng LAN cũng xem được")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ camera (người đã làm mờ)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # tiếng Việt trên console Windows

    log_dir = ROOT / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    load_dotenv(ROOT / ".env")
    source, config = a.source, a.config
    imou = os.getenv("IMOU_RTSP")
    if not source and imou and config in (None, IMOU_CONFIG):
        source, config = encode_password(imou), IMOU_CONFIG
        os.environ["ECOCLASS_SOURCE"] = source       # tiến trình edge đọc biến này, mật khẩu không lộ trên dòng lệnh
        log(f"Camera Imou (từ .env): {hide_password(source)}")
    config = config or DEFAULT_CONFIG
    edge = [py, "scripts/run_edge.py", "--config", config, "--publisher", "http"]
    if a.source:
        edge += ["--source", a.source]
    # Webcam laptop cho ảnh kiểu gương (cấu hình lật ngang); camera điện thoại/IP thì không.
    flip = a.flip or ("none" if source and "://" in source else None)
    if flip:
        edge += ["--flip", flip]
    if a.rotate is not None:
        edge += ["--rotate", str(a.rotate)]
    if a.show:
        edge.append("--show")
    services = [
        Service("server", [py, "-m", "uvicorn", "ecoclass.server.app:app", "--host", a.host, "--port", str(a.port),
                           "--no-access-log"], log_dir),
        Service("edge", edge, log_dir),
    ]
    services[1].next_start = time.time() + 4        # chờ máy chủ sẵn sàng trước khi gửi dữ liệu
    log(f"Dashboard: http://127.0.0.1:{a.port}   (Ctrl+C để tắt)")
    try:
        while True:
            for s in services:
                s.check()
            time.sleep(1)
    except KeyboardInterrupt:
        log("Đang tắt hệ thống...")
    finally:
        for s in reversed(services):
            s.stop()


if __name__ == "__main__":
    main()
