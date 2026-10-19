"""Bot Zalo (Zalo Bot Platform, https://bot.zapps.me/docs): gửi cảnh báo dạng chữ; trả lời lệnh tra cứu.

Cùng giao diện và cùng bộ lệnh với TelegramBot: /status, /alerts, /stats, /ack <alert_id>.
API: POST https://bot-api.zaloplatforms.com/bot<TOKEN>/<method>, chỉ hỗ trợ chat 1-1, tin tối đa 2000 ký tự,
không có định dạng HTML → nội dung (viết cho Telegram) được chuyển sang chữ thường.

Lấy mã chat: nhắn bất kỳ tin nào cho bot → bot trả lời mã chat của bạn → thêm vào ZALO_CHAT_IDS trong .env.
Chỉ người có mã chat trong ZALO_CHAT_IDS mới nhận cảnh báo và dùng được lệnh.
"""
from __future__ import annotations

import html
import re
import threading
import time
from typing import Callable

import requests

API_BASE = "https://bot-api.zaloplatforms.com"
MAX_TEXT = 2000
HELP = "Lệnh: /status /alerts /stats /ack <alert_id>"


def html_to_text(s: str) -> str:
    """Nội dung viết cho Telegram (<b>, &lt;...) → chữ thường cho Zalo."""
    return html.unescape(re.sub(r"</?[a-zA-Z][^>]*>", "", s))


def split_text(s: str, limit: int = MAX_TEXT) -> list[str]:
    """Chia tin dài theo dòng cho vừa giới hạn 2000 ký tự của Zalo."""
    parts, cur = [], ""
    for line in s.split("\n"):
        if len(line) > limit and cur:                 # một dòng quá dài → gửi phần đang gom trước, rồi cắt cứng
            parts.append(cur)
            cur = ""
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def parse_updates(data: dict) -> list[tuple[str, str]]:
    """Phản hồi getUpdates → [(chat_id, text)] của các tin nhắn chữ. ``result`` có thể là một sự kiện hoặc danh sách."""
    if not isinstance(data, dict) or not data.get("ok"):
        return []
    res = data.get("result")
    out = []
    for ev in res if isinstance(res, list) else [res] if res else []:
        msg = (ev or {}).get("message") or {}
        cid = str((msg.get("chat") or {}).get("id") or "")
        text = (msg.get("text") or "").strip()
        if cid and text and not (msg.get("from") or {}).get("is_bot"):
            out.append((cid, text))
    return out


class ZaloBot:
    def __init__(self, token: str | None, chat_ids: list[str],
                 handlers: dict[str, Callable[[str, str], str]] | None = None, api_base: str = API_BASE):
        self.token = token
        self.chat_ids = chat_ids
        self.handlers = handlers or {}
        self.base = f"{api_base.rstrip('/')}/bot{token}" if token else None

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_ids)

    def _call(self, method: str, http_timeout: float = 15, **params) -> dict:
        r = requests.post(f"{self.base}/{method}", json=params, timeout=http_timeout)
        return r.json()

    def _send(self, chat_id: str, text: str) -> bool:
        ok = True
        for part in split_text(html_to_text(text)):
            try:
                res = self._call("sendMessage", chat_id=chat_id, text=part)
                if not res.get("ok"):
                    print("[zalo] gửi thất bại:", res)
                    ok = False
            except (requests.RequestException, ValueError) as e:
                print("[zalo] lỗi gửi:", e)
                ok = False
        return ok

    def send_text(self, text: str) -> bool:
        if not self.enabled:
            print("[zalo:tắt]", html_to_text(text).replace("\n", " | "))
            return False
        results = [self._send(cid, text) for cid in self.chat_ids]
        return any(results)

    def start_polling(self) -> None:
        # Chạy cả khi chưa có ZALO_CHAT_IDS: để người dùng nhắn bot và nhận mã chat của mình.
        if not self.token:
            return
        threading.Thread(target=self._poll, daemon=True, name="zalo").start()

    def _poll(self) -> None:
        try:                                            # getUpdates không chạy khi bot đang đặt webhook
            self._call("deleteWebhook")
        except Exception:
            pass
        print("[zalo] đang nhận tin nhắn (getUpdates)")
        while True:
            try:
                data = self._call("getUpdates", http_timeout=40, timeout="30")   # chờ tin mới tối đa 30 s
                for cid, text in parse_updates(data):
                    known = "" if cid in self.chat_ids else " (chưa có trong ZALO_CHAT_IDS)"
                    print(f"[zalo] tin nhắn từ chat_id={cid}{known}: {text[:50]}")
                    self._send(cid, self.reply(cid, text))
            except requests.Timeout:
                continue                                # không có tin mới trong 30 s
            except Exception as e:
                print("[zalo] polling lỗi:", e)
                time.sleep(5)

    def reply(self, cid: str, text: str) -> str:
        if cid not in self.chat_ids:
            return (f"Mã chat Zalo của bạn: {cid}\n"
                    "Thêm dòng ZALO_CHAT_IDS=" + cid + " vào file .env của EcoClass rồi chạy lại hệ thống "
                    "để nhận cảnh báo.")
        if not text.startswith("/"):
            return HELP
        cmd, _, arg = text[1:].partition(" ")
        handler = self.handlers.get(cmd.lower())
        return handler(arg.strip(), f"zalo:{cid}") if handler else HELP
