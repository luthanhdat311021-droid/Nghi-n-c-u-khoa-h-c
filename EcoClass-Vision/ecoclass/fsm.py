"""Bộ suy luận theo thời gian (máy trạng thái có trễ) cho mỗi phòng học — Hình 2.

Trạng thái:
    OCCUPIED       có người
    EMPTY_PENDING  nghi vắng (N=0, chưa đủ Te)
    ALERT          bỏ quên thiết bị: vắng >= Te và còn thiết bị BẬT
    RESOLVED       đã xử lý: thiết bị đã được tắt (trạng thái tức thời)
    EMPTY_SAFE     trống – an toàn: mọi thiết bị TẮT

Điều kiện "bỏ quên": N(τ)=0 với mọi τ ∈ [t−Te, t] và ∃i: S_i(t)=BẬT.
Điện năng lãng phí: E = Σ P_i · Δt_i, Δt_i là thời gian thiết bị i BẬT khi phòng trống.

Mô-đun này thuần Python, không phụ thuộc OpenCV/YOLO để dễ kiểm thử và dùng lại
trong các thí nghiệm ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .config import FSMConfig

ON, OFF, UNKNOWN = "ON", "OFF", "UNKNOWN"


class RoomState(str, Enum):
    OCCUPIED = "OCCUPIED"
    EMPTY_PENDING = "EMPTY_PENDING"
    ALERT = "ALERT"
    RESOLVED = "RESOLVED"
    EMPTY_SAFE = "EMPTY_SAFE"


EMPTY_STATES = {RoomState.EMPTY_PENDING, RoomState.ALERT, RoomState.RESOLVED, RoomState.EMPTY_SAFE}


@dataclass
class Event:
    type: str                 # state_change | alert | alert_closed
    room_id: str
    ts: float
    state_from: str | None
    state_to: str
    people: int
    devices: dict[str, str]
    devices_on: list[str] = field(default_factory=list)
    empty_seconds: float = 0.0
    waste_wh: float = 0.0
    waste_by_device_wh: dict[str, float] = field(default_factory=dict)
    reason: str | None = None
    alert_id: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class TemporalFSM:
    def __init__(
        self,
        room_id: str,
        cfg: FSMConfig,
        power_w: dict[str, float],
        te_fn: Callable[[float], float] | None = None,
        occupied_confirm_updates: int = 2,
    ):
        self.room_id = room_id
        self.cfg = cfg
        self.power_w = power_w
        self.te_fn = te_fn or (lambda ts: cfg.te_seconds)
        self.occupied_confirm_updates = max(1, occupied_confirm_updates)

        self.state: RoomState | None = None
        self.last_ts: float | None = None
        self.last_person_ts: float | None = None
        self.empty_since: float | None = None
        self.all_off_since: float | None = None
        self.person_streak = 0
        self.alert_started: float | None = None
        self.alert_id: str | None = None
        self.on_seconds: dict[str, float] = {}
        self.alert_count = 0

    # ------------------------------------------------------------------ API
    def update(self, ts: float, people: int, devices: dict[str, str]) -> list[Event]:
        """Cập nhật một quan sát (ts tính bằng giây). Trả về danh sách sự kiện phát sinh."""
        events: list[Event] = []
        dt = 0.0 if self.last_ts is None else max(0.0, ts - self.last_ts)
        self.last_ts = ts

        on = sorted(k for k, v in devices.items() if v == ON)
        self.person_streak = self.person_streak + 1 if people > 0 else 0
        if people > 0:
            self.last_person_ts = ts

        # Tích luỹ thời gian thiết bị BẬT khi phòng trống (dùng trạng thái trước bước này).
        if self.state in EMPTY_STATES and dt > 0:
            for d in on:
                self.on_seconds[d] = self.on_seconds.get(d, 0.0) + dt

        if on:
            self.all_off_since = None
        elif self.all_off_since is None:
            self.all_off_since = ts

        def go(new: RoomState, reason: str | None = None) -> None:
            old = self.state
            self.state = new
            events.append(self._event("state_change", ts, old, new, people, devices, on, reason))

        if self.state is None:
            if people > 0:
                go(RoomState.OCCUPIED, "init")
            else:
                self.empty_since = ts
                go(RoomState.EMPTY_PENDING, "init")
            return events

        person_confirmed = self.person_streak >= self.occupied_confirm_updates

        if self.state == RoomState.OCCUPIED:
            ref = self.last_person_ts if self.last_person_ts is not None else ts
            if people == 0 and ts - ref >= self.cfg.empty_seconds_to_pending:
                self.empty_since = ref
                self.on_seconds = {}
                go(RoomState.EMPTY_PENDING, "no_person")

        elif self.state == RoomState.EMPTY_PENDING:
            if person_confirmed:
                self._reset_empty()
                go(RoomState.OCCUPIED, "person_returned")
            elif ts - self.empty_since >= self.te_fn(ts):
                if on:
                    self._open_alert(ts)
                    go(RoomState.ALERT, "devices_left_on")
                    events.append(self._event("alert", ts, RoomState.EMPTY_PENDING, RoomState.ALERT,
                                              people, devices, on, "devices_left_on"))
                else:
                    go(RoomState.EMPTY_SAFE, "all_off")

        elif self.state == RoomState.ALERT:
            if person_confirmed:
                events.append(self._close_alert(ts, people, devices, on, "person_returned"))
                self._reset_empty()
                go(RoomState.OCCUPIED, "person_returned")
            elif not on and ts - self.all_off_since >= self.cfg.off_confirm_seconds:
                events.append(self._close_alert(ts, people, devices, on, "devices_off"))
                go(RoomState.RESOLVED, "devices_off")

        elif self.state == RoomState.RESOLVED:
            if person_confirmed:
                self._reset_empty()
                go(RoomState.OCCUPIED, "person_returned")
            else:
                go(RoomState.EMPTY_SAFE, "resolved")

        elif self.state == RoomState.EMPTY_SAFE:
            if person_confirmed:
                self._reset_empty()
                go(RoomState.OCCUPIED, "person_returned")
            elif on:
                # Thiết bị bật lại khi phòng vẫn trống (vd. bật từ công tắc hành lang).
                self.on_seconds = {}
                self._open_alert(ts)
                go(RoomState.ALERT, "device_on_while_empty")
                events.append(self._event("alert", ts, RoomState.EMPTY_SAFE, RoomState.ALERT,
                                          people, devices, on, "device_on_while_empty"))
        return events

    def snapshot(self, ts: float, people: int, devices: dict[str, str]) -> dict:
        """Trạng thái hiện tại để gửi heartbeat."""
        on = [k for k, v in devices.items() if v == ON]
        wh = self._waste_wh()
        return {
            "room_id": self.room_id,
            "ts": ts,
            "state": self.state.value if self.state else None,
            "people": people,
            "devices": devices,
            "devices_on": on,
            "empty_seconds": (ts - self.empty_since) if self.state in EMPTY_STATES and self.empty_since else 0.0,
            "te_seconds": self.te_fn(ts),
            "waste_wh_current": round(sum(wh.values()), 3),
            "alert_id": self.alert_id if self.state == RoomState.ALERT else None,
        }

    # ------------------------------------------------------------ internals
    def _open_alert(self, ts: float) -> None:
        self.alert_count += 1
        self.alert_started = ts
        self.alert_id = f"{self.room_id}-{int(ts * 1000)}"

    def _close_alert(self, ts, people, devices, on, reason) -> Event:
        ev = self._event("alert_closed", ts, RoomState.ALERT, RoomState.RESOLVED, people, devices, on, reason)
        self.alert_started = None
        return ev

    def _reset_empty(self) -> None:
        self.empty_since = None
        self.on_seconds = {}

    def _waste_wh(self) -> dict[str, float]:
        return {d: self.power_w.get(d, 0.0) * s / 3600.0 for d, s in self.on_seconds.items()}

    def _event(self, etype, ts, old, new, people, devices, on, reason) -> Event:
        wh = self._waste_wh()
        return Event(
            type=etype,
            room_id=self.room_id,
            ts=ts,
            state_from=old.value if isinstance(old, RoomState) else old,
            state_to=new.value,
            people=people,
            devices=dict(devices),
            devices_on=list(on),
            empty_seconds=(ts - self.empty_since) if self.empty_since is not None else 0.0,
            waste_wh=round(sum(wh.values()), 3),
            waste_by_device_wh={k: round(v, 3) for k, v in wh.items()},
            reason=reason,
            alert_id=self.alert_id,
        )
