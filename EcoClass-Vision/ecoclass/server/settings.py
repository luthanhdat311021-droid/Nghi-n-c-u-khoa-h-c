"""Cấu hình máy chủ qua biến môi trường (có thể đặt trong file .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(Path.cwd() / ".env")


@dataclass
class Settings:
    # PostgreSQL khi triển khai: postgresql+psycopg://user:pass@host/ecoclass
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/ecoclass.db")
    evidence_dir: str = os.getenv("EVIDENCE_DIR", "data/evidence")
    api_key: str | None = os.getenv("ECOCLASS_API_KEY") or None
    mqtt_host: str | None = os.getenv("MQTT_HOST") or None
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_prefix: str = os.getenv("MQTT_PREFIX", "ecoclass")
    # Tầng xác minh 2: off | florence-2-base | florence-2-large
    vlm: str = os.getenv("ECOCLASS_VLM", "off")
    vlm_device: str = os.getenv("ECOCLASS_VLM_DEVICE", "auto")
    timetable: str | None = os.getenv("TIMETABLE_CSV", "configs/timetable.csv")
    tariff_vnd_per_kwh: float = float(os.getenv("TARIFF_VND_PER_KWH", "2500"))
    telegram_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN") or None
    telegram_chat_ids: list[str] = field(
        default_factory=lambda: [c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]
    )
    # Bot Zalo (Zalo Bot Platform): token lấy ở https://bot.zaloplatforms.com; mã chat: nhắn bot một tin bất kỳ
    zalo_token: str | None = os.getenv("ZALO_BOT_TOKEN") or None
    zalo_chat_ids: list[str] = field(
        default_factory=lambda: [c.strip() for c in os.getenv("ZALO_CHAT_IDS", "").split(",") if c.strip()]
    )
    zalo_api_base: str = os.getenv("ZALO_API_BASE", "https://bot-api.zaloplatforms.com")
    public_url: str = os.getenv("PUBLIC_URL", "http://127.0.0.1:8000")


settings = Settings()
