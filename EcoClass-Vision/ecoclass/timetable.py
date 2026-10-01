"""Thời khoá biểu phòng học (CSV) — dùng để chọn ngưỡng vắng Te.

Định dạng CSV (có dòng tiêu đề):
    room_id,weekday,start,end,course
    A1-101,0,07:00,09:30,Lập trình Python
weekday: 0 = Thứ Hai ... 6 = Chủ Nhật.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path


@dataclass(frozen=True)
class Slot:
    room_id: str
    weekday: int
    start: time
    end: time
    course: str = ""


class Timetable:
    def __init__(self, slots: list[Slot] | None = None):
        self.slots = slots or []

    @classmethod
    def load(cls, path: str | Path | None) -> "Timetable":
        if not path or not Path(path).exists():
            return cls([])
        slots = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                slots.append(Slot(
                    room_id=row["room_id"].strip(),
                    weekday=int(row["weekday"]),
                    start=time.fromisoformat(row["start"].strip()),
                    end=time.fromisoformat(row["end"].strip()),
                    course=row.get("course", "").strip(),
                ))
        return cls(slots)

    def current(self, room_id: str, when: datetime) -> Slot | None:
        t = when.time()
        for s in self.slots:
            if s.room_id == room_id and s.weekday == when.weekday() and s.start <= t < s.end:
                return s
        return None

    def in_session(self, room_id: str, when: datetime) -> bool:
        return self.current(room_id, when) is not None

    def for_room(self, room_id: str) -> list[Slot]:
        return [s for s in self.slots if s.room_id == room_id]
