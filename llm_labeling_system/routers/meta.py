from __future__ import annotations

from fastapi import APIRouter

from ..db import get_conn
from ..services.ai_settings import effective_public_settings
from ..services.prompting import LABELS

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/config")
def get_config() -> dict:
    with get_conn() as conn:
        public = effective_public_settings(conn)
    return {
        "deepseek_available": public["has_api_key"],
        "default_model": public["model"],
        "labels": LABELS,
    }


@router.get("/labels-vocab")
async def labels_vocab() -> dict:
    return {"labels": LABELS}
