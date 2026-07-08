from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import repository
from ..db import get_conn
from ..schemas import ScanIn
from ..services import scan_jobs

router = APIRouter(prefix="/api/sessions", tags=["scan"])


def _scan_state(session_id: int) -> dict:
    with get_conn() as conn:
        if repository.get_session_row(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        results = repository.list_flagged_suggestions(conn, session_id)
        stored = repository.count_ai_suggestions(conn, session_id)

    job = scan_jobs.get_job(session_id)
    if job is not None:
        state = scan_jobs.snapshot(job)
    else:
        # No in-memory job (e.g. server restarted): results persist, progress doesn't.
        state = {
            "status": "idle",
            "done": stored,
            "total": stored,
            "errors": 0,
            "source": "",
            "model": "",
            "error_detail": "",
        }
    state["results"] = results
    return state


# Sync handlers on purpose: they touch SQLite and coordinate a worker thread;
# FastAPI runs sync handlers in a thread pool.
@router.post("/{session_id}/scan")
def start_scan(session_id: int, body: ScanIn) -> dict:
    with get_conn() as conn:
        if repository.get_session_row(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    try:
        scan_jobs.start_scan(session_id, mock=body.mock, model_override=body.model)
    except scan_jobs.ScanInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _scan_state(session_id)


@router.get("/{session_id}/scan")
def get_scan(session_id: int) -> dict:
    return _scan_state(session_id)


@router.delete("/{session_id}/scan")
def cancel_scan(session_id: int) -> dict:
    scan_jobs.cancel_scan(session_id)
    return _scan_state(session_id)
