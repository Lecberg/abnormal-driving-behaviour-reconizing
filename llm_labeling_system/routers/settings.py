from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import repository
from ..db import get_conn
from ..schemas import AISettingsIn, AITestIn
from ..services import ai_settings
from ..services.deepseek_client import DeepSeekConfig, test_connection

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_settings() -> dict:
    with get_conn() as conn:
        return ai_settings.effective_public_settings(conn)


# Sync handlers on purpose: they touch SQLite (and /test blocks on HTTP for up
# to 15 s); FastAPI runs sync handlers in a thread pool.
@router.put("")
def put_settings(body: AISettingsIn) -> dict:
    base_url = None if body.base_url is None else body.base_url.strip()
    if base_url and not base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Base URL must start with http:// or https://")
    with get_conn() as conn:
        if body.base_url is not None:
            repository.set_setting(conn, ai_settings.AI_BASE_URL, base_url)
        if body.model is not None:
            repository.set_setting(conn, ai_settings.AI_MODEL, body.model)
        if body.api_key is not None:
            repository.set_setting(conn, ai_settings.AI_API_KEY, body.api_key)
        return ai_settings.effective_public_settings(conn)


@router.post("/test")
def test_settings(body: AITestIn) -> dict:
    with get_conn() as conn:
        stored = ai_settings.resolve_ai_config(conn)
        public = ai_settings.effective_public_settings(conn)
    api_key = (body.api_key or "").strip() or (stored.api_key if stored else "")
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key provided or stored.")
    config = DeepSeekConfig(
        api_key=api_key,
        base_url=(body.base_url or "").strip() or public["base_url"],
        model=(body.model or "").strip() or public["model"],
    )
    ok, detail = test_connection(config)
    # HTTP 200 either way so the dialog can show failures inline.
    return {"ok": ok, "detail": detail, "model": config.model}
