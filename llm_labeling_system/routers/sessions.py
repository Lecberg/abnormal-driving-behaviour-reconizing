from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import repository
from ..config import ensure_dirs, settings
from ..db import get_conn
from ..services import scan_jobs
from ..services.windowing import WindowConfig, generate_windows, load_gps_data

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions() -> dict:
    with get_conn() as conn:
        return {"sessions": repository.list_sessions(conn)}


# Sync handler on purpose: pandas parsing blocks; FastAPI runs sync handlers in
# a thread pool so the event loop stays responsive.
@router.post("")
def create_session(
    file: UploadFile = File(...),
    window_size: int = Form(10),
    stride: int = Form(5),
    max_rows: Optional[int] = Form(None),
    max_windows: int = Form(500),
    project: Optional[str] = Form(None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please select a CSV or XLSX file.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported.")
    if window_size < 2:
        raise HTTPException(status_code=400, detail="Window size must be at least 2.")
    if stride < 1:
        raise HTTPException(status_code=400, detail="Stride must be at least 1.")
    if max_windows < 1:
        raise HTTPException(status_code=400, detail="Max windows must be at least 1.")

    ensure_dirs()
    target = settings.upload_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"

    # Stream to disk with a hard size cap so a huge upload can't fill the disk.
    limit = settings.max_upload_bytes
    written = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise ValueError("upload too large")
                out.write(chunk)
    except ValueError:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
        )

    try:
        effective_max_rows = max_rows if max_rows is not None else settings.max_rows
        df = load_gps_data(target, max_rows=effective_max_rows)
        windows = generate_windows(
            df,
            WindowConfig(window_size=window_size, stride=stride),
            max_windows=max_windows,
        )
    except Exception as exc:  # noqa: BLE001 - return a clear browser error.
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read this file: {exc}") from exc

    if not windows:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No windows generated. Each vehicle needs at least window_size rows.",
        )

    with get_conn() as conn:
        project_id = repository.get_or_create_project(conn, project.strip()) if project and project.strip() else None
        session_id = repository.create_session(
            conn,
            source_name=Path(file.filename).name,
            source_path=str(target),
            window_size=window_size,
            stride=stride,
            max_rows=effective_max_rows,
            max_windows=max_windows,
            row_count=int(len(df)),
            windows=windows,
            project_id=project_id,
        )
        return repository.get_session(conn, session_id)


@router.get("/{session_id}")
async def get_session(session_id: int) -> dict:
    with get_conn() as conn:
        session = repository.get_session(conn, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session


@router.delete("/{session_id}")
def delete_session(session_id: int) -> dict:
    # Stop any running scan promptly; the worker also self-stops when the
    # session's windows disappear.
    scan_jobs.cancel_scan(session_id)
    with get_conn() as conn:
        source_path = repository.delete_session(conn, session_id)
    if source_path is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if source_path:
        try:
            Path(source_path).unlink(missing_ok=True)
        except OSError:
            pass
    return {"ok": True}
