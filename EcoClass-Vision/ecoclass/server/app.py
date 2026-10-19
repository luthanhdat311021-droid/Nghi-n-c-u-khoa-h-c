"""Máy chủ EcoClass-Vision (FastAPI).

- Nhận heartbeat/sự kiện từ thiết bị biên qua MQTT (Mosquitto) hoặc HTTP.
- Lưu trữ vào PostgreSQL/SQLite; lưu ảnh bằng chứng đã làm mờ (không lưu video).
- Gọi tầng xác minh 2 (Florence-2) cho mỗi cảnh báo, rồi gửi Telegram.
- Cung cấp API + dashboard web: trạng thái phòng, lịch sử cảnh báo, thống kê kWh/chi phí.

Chạy:  uvicorn ecoclass.server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import html
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sqlalchemy import select

from ..edge.privacy import blur_boxes
from ..timetable import Timetable
from .db import Alert, Event, Room, SensorLog, SessionLocal
from .settings import settings
from .telegram_bot import STATE_VI, TelegramBot, devices_vi
from .zalo_bot import ZaloBot
from .verifier import make_verifier

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="EcoClass-Vision", version="0.1.0")
verifier = make_verifier(settings.vlm, settings.vlm_device)
timetable = Timetable.load(settings.timetable)
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verify")
Path(settings.evidence_dir).mkdir(parents=True, exist_ok=True)
SENSOR_LOG_SECONDS = 30.0          # ghi lịch sử nhiệt độ thưa để CSDL không phình to
OFFLINE_SECONDS = 30.0             # quá lâu không nhận heartbeat → thiết bị biên mất kết nối
_last_sensor_log: dict[str, float] = {}
_frames: dict[str, tuple[bytes, float]] = {}   # room_id → (JPEG mới nhất, thời điểm nhận)


def check_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "Sai API key")


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts))


# ------------------------------------------------------------------ ingest
def handle_state(p: dict) -> None:
    with SessionLocal() as s:
        room = s.get(Room, p["room_id"]) or Room(room_id=p["room_id"])
        room.state = p.get("state")
        room.people = int(p.get("people", 0))
        room.devices = p.get("devices", {})
        room.empty_seconds = float(p.get("empty_seconds", 0.0))
        room.waste_wh_current = float(p.get("waste_wh_current", 0.0))
        room.dark = bool(p.get("dark", False))
        room.last_seen = _dt(p["ts"])
        room.fps = p.get("fps")
        sensors = p.get("sensors")
        if sensors is not None:
            room.sensors = sensors
            ts = float(p["ts"])
            if sensors.get("temperature") is not None and \
                    ts - _last_sensor_log.get(room.room_id, 0.0) >= SENSOR_LOG_SECONDS:
                _last_sensor_log[room.room_id] = ts
                s.add(SensorLog(room_id=room.room_id, time=room.last_seen,
                                temperature=sensors.get("temperature"), humidity=sensors.get("humidity")))
        s.merge(room)
        s.commit()


def handle_event(p: dict) -> None:
    jpeg_b64 = p.pop("evidence_jpeg_b64", None)
    when = _dt(p["ts"])
    with SessionLocal() as s:
        room = s.get(Room, p["room_id"]) or Room(room_id=p["room_id"])
        room.state = p["state_to"]
        room.people = int(p.get("people", 0))
        room.devices = p.get("devices", {})
        room.last_seen = when
        s.merge(room)
        s.add(Event(room_id=p["room_id"], type=p["type"], time=when, state_from=p.get("state_from"),
                    state_to=p["state_to"], people=int(p.get("people", 0)), devices=p.get("devices", {}),
                    devices_on=p.get("devices_on", []), waste_wh=float(p.get("waste_wh", 0.0)),
                    reason=p.get("reason"), alert_id=p.get("alert_id")))

        if p["type"] == "alert":
            path = None
            if jpeg_b64:
                path = str(Path(settings.evidence_dir) / f"{p['alert_id']}.jpg")
                Path(path).write_bytes(base64.b64decode(jpeg_b64))
            slot = timetable.current(p["room_id"], when)
            s.merge(Alert(alert_id=p["alert_id"], room_id=p["room_id"], started=when,
                          devices_on=p.get("devices_on", []), power_w=p.get("power_w", {}),
                          empty_seconds=float(p.get("empty_seconds", 0.0)), in_class=slot is not None,
                          course=slot.course if slot else None, temperature=p.get("temperature"),
                          status="verifying" if settings.vlm != "off" else "open", evidence_path=path))
            s.commit()
            worker.submit(_verify_and_notify, p["alert_id"])
        elif p["type"] == "alert_closed":
            a = s.get(Alert, p["alert_id"])
            if a is not None:
                a.ended = when
                a.close_reason = p.get("reason")
                a.waste_wh = float(p.get("waste_wh", 0.0))
                a.waste_by_device_wh = p.get("waste_by_device_wh", {})
                tariff = float(p.get("tariff_vnd_per_kwh", settings.tariff_vnd_per_kwh))
                a.cost_vnd = round(a.waste_wh / 1000.0 * tariff, 0)
                if a.status != "rejected":
                    a.status = "closed"
                s.commit()
                if a.notified:
                    mins = (a.ended - a.started).total_seconds() / 60
                    reason = "thiết bị đã được tắt" if a.close_reason == "devices_off" else "có người quay lại phòng"
                    bot.send_text(f"✅ <b>{html.escape(a.room_id)}</b>: {reason} sau {mins:.0f} phút.\n"
                                  f"Lãng phí ước tính: {a.waste_wh:.0f} Wh ≈ {a.cost_vnd:,.0f} đ")
            else:
                s.commit()
        else:
            s.commit()


def _verify_and_notify(alert_id: str) -> None:
    with SessionLocal() as s:
        a = s.get(Alert, alert_id)
        if a is None:
            return
        img = cv2.imread(a.evidence_path) if a.evidence_path else None
        if img is not None:
            try:
                res = verifier.verify(img)
            except Exception as e:  # lỗi tầng 2 không được chặn cảnh báo
                print("[verifier] lỗi:", e)
                res = None
                a.verify_note = f"lỗi xác minh: {e}"
            if res is not None:
                a.verified = not res.person_present
                a.verify_note = res.caption[:500]
                if res.person_present:
                    a.status = "rejected"
                    if res.person_boxes:  # làm mờ người mà tầng 1 bỏ sót trước khi lưu
                        cv2.imwrite(a.evidence_path, blur_boxes(img, res.person_boxes))
                elif a.status == "verifying":
                    a.status = "open"
        if a.status == "verifying":
            a.status = "open"
        s.commit()
        if a.status == "open":
            a.notified = _notify_alert(a)
            s.commit()


def _notify_alert(a: Alert) -> bool:
    devices = devices_vi(a.devices_on) or "?"
    kw = sum(a.power_w.values()) / 1000 if a.power_w else 0
    ctx = f"Lịch học: {html.escape(a.course)}" if a.in_class and a.course else "Ngoài giờ học"
    temp = f"Nhiệt độ phòng: {a.temperature:.1f} °C\n" if a.temperature is not None else ""
    caption = (f"⚠️ <b>Phòng {html.escape(a.room_id)} bị bỏ quên thiết bị điện</b>\n"
               f"Thời điểm: {a.started:%H:%M %d/%m/%Y}\nThiết bị đang bật: {html.escape(devices)} (~{kw:.2f} kW)\n{temp}"
               f"Không có người trong {a.empty_seconds / 60:.0f} phút. {ctx}\n"
               f"Xác nhận: /ack {a.alert_id}")
    zalo_ok = zalo.send_text(caption)              # Zalo: chỉ gửi chữ (ảnh bằng chứng cần URL công khai)
    jpeg = Path(a.evidence_path).read_bytes() if a.evidence_path and Path(a.evidence_path).exists() else None
    if jpeg:
        return bot.send_photo(caption, jpeg) or zalo_ok
    bot.send_text(caption)
    return bot.enabled or zalo_ok


# ------------------------------------------------------------------ Telegram commands
def _cmd_status(arg: str, cid: str) -> str:
    with SessionLocal() as s:
        rooms = s.scalars(select(Room).order_by(Room.room_id)).all()
    if not rooms:
        return "Chưa có phòng nào gửi dữ liệu."
    lines = []
    for r in rooms:
        on = [k for k, v in (r.devices or {}).items() if v == "ON"]
        t = (r.sensors or {}).get("temperature")
        temp = f" · {t:.1f} °C" if t is not None else ""
        lines.append(f"<b>{html.escape(r.room_id)}</b>: {STATE_VI.get(r.state, r.state)} · {r.people} người · "
                     f"bật: {devices_vi(on) or 'không'}{temp}")
    return "\n".join(lines)


def _cmd_alerts(arg: str, cid: str) -> str:
    with SessionLocal() as s:
        rows = s.scalars(select(Alert).where(Alert.status == "open").order_by(Alert.started.desc())).all()
    return "\n".join(f"• {a.room_id} từ {a.started:%H:%M} — {devices_vi(a.devices_on)} (/ack {a.alert_id})" for a in rows) \
        or "Không có cảnh báo đang mở."


def _cmd_stats(arg: str, cid: str) -> str:
    st = compute_stats(7)
    lines = [f"<b>7 ngày qua</b>: {st['total_alerts']} cảnh báo, {st['total_kwh']:.2f} kWh ≈ {st['total_cost_vnd']:,.0f} đ"]
    lines += [f"• {r['room_id']}: {r['alerts']} lần, {r['kwh']:.2f} kWh" for r in st["by_room"]]
    return "\n".join(lines)


def _cmd_ack(arg: str, cid: str) -> str:
    return ack_alert(arg, cid if cid.startswith("zalo:") else f"telegram:{cid}")


COMMANDS = {"status": _cmd_status, "alerts": _cmd_alerts, "stats": _cmd_stats, "ack": _cmd_ack, "start": _cmd_status}
bot = TelegramBot(settings.telegram_token, settings.telegram_chat_ids, COMMANDS)
zalo = ZaloBot(settings.zalo_token, settings.zalo_chat_ids, COMMANDS, api_base=settings.zalo_api_base)


@app.on_event("startup")
def _startup() -> None:
    bot.start_polling()
    zalo.start_polling()
    if settings.mqtt_host:
        from .mqtt_bridge import start_mqtt_bridge
        start_mqtt_bridge(settings.mqtt_host, settings.mqtt_port, settings.mqtt_prefix, handle_state, handle_event,
                          settings.api_key)


# ------------------------------------------------------------------ REST API
@app.post("/api/ingest/state", dependencies=[Depends(check_key)])
def ingest_state(p: dict):
    handle_state(p)
    return {"ok": True}


@app.post("/api/ingest/event", dependencies=[Depends(check_key)])
def ingest_event(p: dict):
    handle_event(p)
    return {"ok": True}


@app.post("/api/ingest/frame/{room_id}", dependencies=[Depends(check_key)])
async def ingest_frame(room_id: str, request: Request):
    """Ảnh xem trực tiếp mới nhất của phòng (đã làm mờ người tại biên). Chỉ giữ trong RAM, không lưu đĩa."""
    data = await request.body()
    if not data.startswith(b"\xff\xd8"):
        raise HTTPException(400, "Cần ảnh JPEG")
    _frames[room_id] = (data, time.time())
    return {"ok": True}


@app.get("/api/rooms/{room_id}/frame.jpg")
def room_frame(room_id: str):
    f = _frames.get(room_id)
    if f is None or time.time() - f[1] > OFFLINE_SECONDS:
        raise HTTPException(404, "Chưa có ảnh camera")
    return Response(f[0], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/rooms/{room_id}/stream.mjpg")
async def room_stream(room_id: str, request: Request):
    """Video trực tiếp dạng MJPEG (dùng thẳng trong thẻ <img>), gửi mỗi ảnh mới ngay khi thiết bị biên đẩy lên."""
    async def gen():
        last = None
        while not await request.is_disconnected():
            f = _frames.get(room_id)
            if f is not None and f[1] != last:
                last = f[1]
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f[0])).encode()
                       + b"\r\n\r\n" + f[0] + b"\r\n")
            await asyncio.sleep(0.01)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/rooms")
def rooms():
    with SessionLocal() as s:
        rows = s.scalars(select(Room).order_by(Room.room_id)).all()
        open_alerts = {a.room_id: a.alert_id for a in s.scalars(select(Alert).where(Alert.status == "open"))}
    now = datetime.now()
    return [{
        "room_id": r.room_id, "state": r.state, "state_vi": STATE_VI.get(r.state, r.state), "people": r.people,
        "devices": r.devices, "empty_seconds": r.empty_seconds, "waste_wh_current": r.waste_wh_current,
        "dark": r.dark, "last_seen": r.last_seen.isoformat() if r.last_seen else None,
        "online": bool(r.last_seen and (now - r.last_seen).total_seconds() <= OFFLINE_SECONDS),
        "sensors": r.sensors, "fps": r.fps, "open_alert": open_alerts.get(r.room_id),
        "has_frame": r.room_id in _frames and time.time() - _frames[r.room_id][1] <= OFFLINE_SECONDS,
    } for r in rows]


@app.get("/api/sensors")
def sensor_history(room_id: str, minutes: int = Query(120, ge=1, le=60 * 24 * 7)):
    since = datetime.now() - timedelta(minutes=minutes)
    with SessionLocal() as s:
        q = select(SensorLog).where(SensorLog.room_id == room_id, SensorLog.time >= since).order_by(SensorLog.time)
        return [{"time": x.time.isoformat(), "temperature": x.temperature, "humidity": x.humidity}
                for x in s.scalars(q)]


def _alert_dict(a: Alert) -> dict:
    return {
        "alert_id": a.alert_id, "room_id": a.room_id, "started": a.started.isoformat(),
        "ended": a.ended.isoformat() if a.ended else None,
        "duration_min": round(((a.ended or datetime.now()) - a.started).total_seconds() / 60, 1),
        "devices_on": a.devices_on, "status": a.status, "close_reason": a.close_reason,
        "waste_wh": a.waste_wh, "cost_vnd": a.cost_vnd, "in_class": a.in_class, "course": a.course,
        "verified": a.verified, "verify_note": a.verify_note, "has_evidence": bool(a.evidence_path),
        "acknowledged_by": a.acknowledged_by, "temperature": a.temperature,
    }


@app.get("/api/alerts")
def alerts(limit: int = Query(50, le=500), room_id: str | None = None, status: str | None = None):
    with SessionLocal() as s:
        q = select(Alert).order_by(Alert.started.desc()).limit(limit)
        if room_id:
            q = q.where(Alert.room_id == room_id)
        if status:
            q = q.where(Alert.status == status)
        return [_alert_dict(a) for a in s.scalars(q)]


@app.get("/api/alerts/{alert_id}/evidence")
def evidence(alert_id: str):
    with SessionLocal() as s:
        a = s.get(Alert, alert_id)
    if a is None or not a.evidence_path or not Path(a.evidence_path).exists():
        raise HTTPException(404)
    return FileResponse(a.evidence_path, media_type="image/jpeg")


def ack_alert(alert_id: str, who: str) -> str:
    with SessionLocal() as s:
        a = s.get(Alert, alert_id)
        if a is None:
            return "Không tìm thấy cảnh báo."
        a.acknowledged_by, a.acknowledged_at = who, datetime.now()
        s.commit()
    return f"Đã ghi nhận xử lý cảnh báo {alert_id}."


@app.post("/api/alerts/{alert_id}/ack")
def ack(alert_id: str, who: str = "dashboard"):
    return {"message": ack_alert(alert_id, who)}


@app.get("/api/events")
def events(limit: int = Query(100, le=1000), room_id: str | None = None):
    with SessionLocal() as s:
        q = select(Event).order_by(Event.time.desc()).limit(limit)
        if room_id:
            q = q.where(Event.room_id == room_id)
        return [{"time": e.time.isoformat(), "room_id": e.room_id, "type": e.type, "from": e.state_from,
                 "to": e.state_to, "people": e.people, "devices_on": e.devices_on, "reason": e.reason,
                 "waste_wh": e.waste_wh} for e in s.scalars(q)]


def compute_stats(days: int = 7) -> dict:
    since = datetime.now() - timedelta(days=days)
    with SessionLocal() as s:
        rows = s.scalars(select(Alert).where(Alert.started >= since)).all()
    valid = [a for a in rows if a.status != "rejected"]
    by_room: dict[str, dict] = defaultdict(lambda: {"alerts": 0, "kwh": 0.0, "cost_vnd": 0.0, "minutes": 0.0})
    by_hour = [0.0] * 24
    by_day: dict[str, float] = defaultdict(float)
    by_device: dict[str, float] = defaultdict(float)
    for a in valid:
        r = by_room[a.room_id]
        r["alerts"] += 1
        r["kwh"] += a.waste_wh / 1000
        r["cost_vnd"] += a.cost_vnd
        r["minutes"] += ((a.ended or datetime.now()) - a.started).total_seconds() / 60
        by_hour[a.started.hour] += a.waste_wh / 1000
        by_day[a.started.strftime("%Y-%m-%d")] += a.waste_wh / 1000
        for d, wh in (a.waste_by_device_wh or {}).items():
            by_device[d] += wh / 1000
    n_days = max(1, days)
    n_rooms = max(1, len({a.room_id for a in rows}))
    return {
        "days": days,
        "total_alerts": len(valid),
        "rejected_by_vlm": len(rows) - len(valid),
        "false_alarms_per_room_day": round((len(rows) - len(valid)) / n_rooms / n_days, 3),
        "total_kwh": round(sum(r["kwh"] for r in by_room.values()), 3),
        "total_cost_vnd": round(sum(r["cost_vnd"] for r in by_room.values())),
        "by_room": [{"room_id": k, **{kk: round(vv, 3) for kk, vv in v.items()}} for k, v in sorted(by_room.items())],
        "by_hour_kwh": [round(x, 4) for x in by_hour],
        "by_day_kwh": [{"date": k, "kwh": round(v, 4)} for k, v in sorted(by_day.items())],
        "by_device_kwh": {k: round(v, 4) for k, v in sorted(by_device.items())},
    }


@app.get("/api/stats")
def stats(days: int = Query(7, ge=1, le=365)):
    return compute_stats(days)


@app.get("/api/timetable")
def get_timetable(room_id: str | None = None):
    slots = timetable.for_room(room_id) if room_id else timetable.slots
    return [{"room_id": x.room_id, "weekday": x.weekday, "start": x.start.isoformat("minutes"),
             "end": x.end.isoformat("minutes"), "course": x.course} for x in slots]


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/cameras", response_class=HTMLResponse)
def cameras_page():
    return (STATIC / "cameras.html").read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    return {"ok": True, "vlm": settings.vlm, "telegram": bot.enabled, "zalo": zalo.enabled, "mqtt": bool(settings.mqtt_host)}
