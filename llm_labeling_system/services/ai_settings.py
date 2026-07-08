"""Effective AI provider configuration: DB-stored settings win, env vars are the fallback.

The frozen ``config.settings`` singleton stays env-only; this module layers the
mutable ``app_settings`` table on top of it.
"""

from __future__ import annotations

import os
from sqlite3 import Connection
from typing import Any, Optional

from .. import repository
from ..config import settings
from .deepseek_client import DeepSeekConfig

AI_API_KEY = "ai_api_key"
AI_BASE_URL = "ai_base_url"
AI_MODEL = "ai_model"


def _env_api_key() -> str:
    # Read at call time (not import time) so tests can monkeypatch the env.
    return os.getenv("DEEPSEEK_API_KEY", "").strip()


def resolve_ai_config(
    conn: Connection, model_override: Optional[str] = None
) -> Optional[DeepSeekConfig]:
    """Build the client config from DB-else-env values; None when no API key exists."""
    api_key = repository.get_setting(conn, AI_API_KEY) or _env_api_key()
    if not api_key:
        return None
    base_url = repository.get_setting(conn, AI_BASE_URL) or settings.deepseek_base_url
    model = model_override or repository.get_setting(conn, AI_MODEL) or settings.deepseek_model
    return DeepSeekConfig(api_key=api_key, base_url=base_url, model=model)


def effective_public_settings(conn: Connection) -> dict[str, Any]:
    """Masked settings view for the browser — never includes the API key itself."""
    db_key = repository.get_setting(conn, AI_API_KEY)
    env_key = _env_api_key()
    return {
        "base_url": repository.get_setting(conn, AI_BASE_URL) or settings.deepseek_base_url,
        "model": repository.get_setting(conn, AI_MODEL) or settings.deepseek_model,
        "has_api_key": bool(db_key or env_key),
        "api_key_source": "db" if db_key else ("env" if env_key else None),
    }
