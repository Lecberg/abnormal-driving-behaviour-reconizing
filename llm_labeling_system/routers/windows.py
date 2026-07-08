from __future__ import annotations

from sqlite3 import Connection
from typing import Any

from fastapi import APIRouter, HTTPException

from .. import repository
from ..db import get_conn

router = APIRouter(prefix="/api/sessions", tags=["windows"])


def _attach(conn: Connection, session_id: int, window: dict[str, Any]) -> dict[str, Any]:
    label_obj = repository.get_label(conn, session_id, window["window_pk"])
    return {
        "index": window["seq"],
        "total": repository.window_count(conn, session_id),
        "window": window,
        "label": {"label": label_obj} if label_obj else None,
        "progress": repository.session_progress(conn, session_id),
    }


@router.get("/{session_id}/windows/{seq}")
async def get_window(session_id: int, seq: int) -> dict:
    with get_conn() as conn:
        if repository.get_session_row(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        window = repository.get_window(conn, session_id, seq)
        if not window:
            raise HTTPException(status_code=404, detail="Window index is out of range.")
        return _attach(conn, session_id, window)


@router.get("/{session_id}/next-unlabeled")
async def next_unlabeled(session_id: int) -> dict:
    with get_conn() as conn:
        if repository.get_session_row(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        seq = repository.next_unlabeled_seq(conn, session_id)
        if seq is None:
            return {"done": True, "progress": repository.session_progress(conn, session_id)}
        window = repository.get_window(conn, session_id, seq)
        return _attach(conn, session_id, window)
