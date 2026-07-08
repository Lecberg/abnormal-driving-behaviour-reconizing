from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .. import repository
from ..config import settings
from ..db import get_conn
from ..schemas import LabelIn
from ..services.export import export_csv, export_jsonl
from ..services.prompting import LABELS

router = APIRouter(prefix="/api/sessions", tags=["labels"])

ALLOWED_LABELS = set(LABELS)


def _risk_for_label(label: str) -> str:
    if label == "normal":
        return "low"
    if label == "unclear":
        return "unclear"
    return "medium"


@router.post("/{session_id}/labels")
def save_label(session_id: int, body: LabelIn) -> dict:
    if body.label not in ALLOWED_LABELS:
        raise HTTPException(status_code=400, detail="Invalid label.")
    with get_conn() as conn:
        window = repository.get_window(conn, session_id, body.seq)
        if window is None:
            raise HTTPException(status_code=404, detail="Window not found in this session.")

        confidence = max(0.0, min(1.0, float(body.confidence)))
        use_for_training = body.use_for_training
        if use_for_training is None:
            use_for_training = body.label != "unclear" and confidence >= 0.55
        if body.label == "unclear" or confidence < 0.55:
            use_for_training = False

        flags = window.get("summary", {}).get("data_quality_flags", [])
        repository.upsert_label(conn, session_id, window["window_pk"], {
            "label": body.label,
            "confidence": round(confidence, 4),
            "risk_level": _risk_for_label(body.label),
            "use_for_training": use_for_training,
            "human_review_needed": False,
            "reason": body.notes.strip(),
            "evidence": ["manual_label"],
            "data_quality_flags": flags,
            "source": "manual",
            "model": "human",
        })
        return {"ok": True, "progress": repository.session_progress(conn, session_id)}


@router.get("/{session_id}/labels")
async def list_labels(
    session_id: int,
    label: str = Query("all"),
    min_confidence: float = Query(0.0),
    training_only: bool = Query(False),
) -> dict:
    with get_conn() as conn:
        items = repository.list_labels(conn, session_id, label, min_confidence, training_only)
        return {"count": len(items), "items": items}


@router.get("/{session_id}/export")
def export_session(
    session_id: int, fmt: str = Query("csv"), source: str = Query("labels")
) -> FileResponse:
    if source not in {"labels", "ai"}:
        raise HTTPException(status_code=400, detail="source must be 'labels' or 'ai'.")
    with get_conn() as conn:
        if repository.get_session_row(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    stem = "ai_scan_labels" if source == "ai" else "manual_labels"
    if fmt == "jsonl":
        path = export_jsonl(
            settings.export_dir / f"session_{session_id}_{stem}.jsonl", session_id, table=source
        )
        return FileResponse(path, filename=f"{stem}.jsonl")
    path = export_csv(
        settings.export_dir / f"session_{session_id}_{stem}.csv", session_id, table=source
    )
    return FileResponse(path, filename=f"{stem}.csv")
