from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import repository
from ..db import get_conn
from ..schemas import SuggestIn
from ..services.ai_settings import resolve_ai_config
from ..services.deepseek_client import DeepSeekLabeler
from ..services.prompting import mock_label

router = APIRouter(prefix="/api/sessions", tags=["suggest"])


# Sync handler on purpose: the LLM HTTP call blocks for up to minutes and
# must not stall the event loop. FastAPI runs sync handlers in a thread pool.
@router.post("/{session_id}/suggest")
def suggest(session_id: int, body: SuggestIn) -> dict:
    with get_conn() as conn:
        window = repository.get_window(conn, session_id, body.seq)
        config = None if body.mock else resolve_ai_config(conn, model_override=body.model)
    if window is None:
        raise HTTPException(status_code=404, detail="Window not found in this session.")

    if config is None:
        return {
            "ok": True,
            "source": "mock",
            "model": "mock",
            "seq": body.seq,
            "suggestion": mock_label(window),
        }

    try:
        suggestion = DeepSeekLabeler(config).label_window(window)
    except Exception as exc:  # noqa: BLE001 - show a clear browser error.
        raise HTTPException(status_code=502, detail=f"AI suggestion failed: {exc}") from exc

    return {
        "ok": True,
        "source": "ai",
        "model": config.model,
        "seq": body.seq,
        "suggestion": suggestion,
    }
