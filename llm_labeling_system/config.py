from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_opt_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, resolved once from environment variables."""

    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    data_dir: Path
    db_path: Path
    upload_dir: Path
    export_dir: Path
    max_upload_mb: int
    max_rows: int | None

    @property
    def deepseek_available(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    data_dir = Path(os.getenv("LABELING_DATA_DIR", str(PACKAGE_DIR / "data"))).resolve()
    db_path = Path(os.getenv("LABELING_DB_PATH", str(data_dir / "labeling.db"))).resolve()
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        data_dir=data_dir,
        db_path=db_path,
        upload_dir=data_dir / "uploads",
        export_dir=data_dir / "exports",
        max_upload_mb=_env_int("LABELING_MAX_UPLOAD_MB", 200),
        max_rows=_env_opt_int("LABELING_MAX_ROWS"),
    )


settings = load_settings()


def ensure_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.export_dir.mkdir(parents=True, exist_ok=True)
