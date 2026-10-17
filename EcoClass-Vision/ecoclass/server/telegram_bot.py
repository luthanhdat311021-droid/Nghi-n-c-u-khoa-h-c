"""Bot Telegram: gửi cảnh báo kèm ảnh bằng chứng đã làm mờ; trả lời lệnh tra cứu.

Lệnh: /status — trạng thái các phòng, /alerts — cảnh báo đang mở, /stats — thống kê 7 ngày,
       /ack <alert_id> — xác nhận đã xử lý.
Chỉ phản hồi các chat_id trong TELEGRAM_CHAT_IDS.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import requests

STATE_VI = {
    "OCCUPIED": "Có người", "EMPTY_PENDING": "Nghi vắng", "ALERT": "BỎ QUÊN THIẾT BỊ",
    "RESOLVED": "Đã xử lý", "EMPTY_SAFE": "Trống – an toàn", None: "Chưa rõ",
}
DEVICE_VI = {"lights": "đèn", "fans": "quạt", "ac": "máy lạnh (nhiệt độ < 30 °C)"}


def devices_vi(ids) -> str:
    return ", ".join(DEVICE_VI.get(d, d) for d in ids)


class TelegramBot:
    def __init__(self, token: str | None, chat_ids: list[str], handlers: dict[str, Callable[[str, str], str]] | None = None):
        self.token = token
        self.chat_ids = chat_ids
        self.handlers = handlers or {}
        self.base = f"https://api.telegram.org/bot{token}" if token else None
        self._offset = 0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_ids)

    def send_text(self, text: str) -> None:
        if not self.enabled:
            print("[telegram:tắt]", text.replace("\n", " | "))
            return
        for cid in self.chat_ids:
            try:
                requests.post(f"{self.base}/sendMessage", data={"chat_id": cid, "text": text, "parse_mode": "HTML"}, timeout=15)
            except requests.RequestException as e:
                print("[telegram] lỗi gửi:", e)

    def send_photo(self, caption: str, jpeg: bytes) -> bool:
        if not self.enabled:
            print("[telegram:tắt]", caption.replace("\n", " | "))
            return False
        ok = False
        for cid in self.chat_ids:
            try:
                r = requests.post(f"{self.base}/sendPhoto", data={"chat_id": cid, "caption": caption, "parse_mode": "HTML"},
                                  files={"photo": ("evidence.jpg", jpeg, "image/jpeg")}, timeout=30)
                ok = ok or r.ok
            except requests.RequestException as e:
                print("[telegram] lỗi gửi ảnh:", e)
        return ok

    def start_polling(self) -> None:
        if not self.enabled:
            return
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        while True:
            try:
                r = requests.get(f"{self.base}/getUpdates", params={"timeout": 30, "offset": self._offset}, timeout=40)
                for upd in r.json().get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    cid = str(msg.get("chat", {}).get("id", ""))
                    text = (msg.get("text") or "").strip()
                    if cid not in self.chat_ids or not text.startswith("/"):
                        continue
                    cmd, _, arg = text[1:].partition(" ")
                    handler = self.handlers.get(cmd.split("@")[0].lower())
                    reply = handler(arg.strip(), cid) if handler else "Lệnh: /status /alerts /stats /ack &lt;alert_id&gt;"
                    requests.post(f"{self.base}/sendMessage", data={"chat_id": cid, "text": reply, "parse_mode": "HTML"}, timeout=15)
            except Exception as e:
                print("[telegram] polling lỗi:", e)
                time.sleep(5)
