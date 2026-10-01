"""Mô hình dữ liệu (SQLAlchemy 2) — chạy được với PostgreSQL (triển khai) hoặc SQLite (thử nghiệm)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .settings import settings


class Base(DeclarativeBase):
    pass


class Room(Base):
    __tablename__ = "rooms"
    room_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[str | None] = mapped_column(String(32))
    people: Mapped[int] = mapped_column(Integer, default=0)
    devices: Mapped[dict] = mapped_column(JSON, default=dict)
    empty_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    waste_wh_current: Mapped[float] = mapped_column(Float, default=0.0)
    dark: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(32), index=True)
    time: Mapped[datetime] = mapped_column(DateTime, index=True)
    state_from: Mapped[str | None] = mapped_column(String(32))
    state_to: Mapped[str] = mapped_column(String(32))
    people: Mapped[int] = mapped_column(Integer, default=0)
    devices: Mapped[dict] = mapped_column(JSON, default=dict)
    devices_on: Mapped[list] = mapped_column(JSON, default=list)
    waste_wh: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(String(64))
    alert_id: Mapped[str | None] = mapped_column(String(96), index=True)


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    room_id: Mapped[str] = mapped_column(String(64), ForeignKey("rooms.room_id"), index=True)
    started: Mapped[datetime] = mapped_column(DateTime, index=True)
    ended: Mapped[datetime | None] = mapped_column(DateTime)
    devices_on: Mapped[list] = mapped_column(JSON, default=list)
    power_w: Mapped[dict] = mapped_column(JSON, default=dict)
    empty_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    in_class: Mapped[bool] = mapped_column(Boolean, default=False)
    course: Mapped[str | None] = mapped_column(String(128))
    # open | closed | rejected (tầng 2 phát hiện còn người → cảnh báo giả)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    close_reason: Mapped[str | None] = mapped_column(String(64))
    waste_wh: Mapped[float] = mapped_column(Float, default=0.0)
    waste_by_device_wh: Mapped[dict] = mapped_column(JSON, default=dict)
    cost_vnd: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_path: Mapped[str | None] = mapped_column(Text)
    verified: Mapped[bool | None] = mapped_column(Boolean)
    verify_note: Mapped[str | None] = mapped_column(Text)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime)
    notified: Mapped[bool] = mapped_column(Boolean, default=False)


def make_engine(url: str | None = None):
    url = url or settings.database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
