"""In-process background scan jobs: label every window of a session with the
configured LLM (or the deterministic mock) and persist the verdicts into the
``ai_suggestions`` table, which never touches human labels.

Single-user tool: one running job per session, plain threads, no queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .. import repository
from ..db import get_conn
from .ai_settings import resolve_ai_config
from .deepseek_client import DeepSeekConfig, DeepSeekLabeler
from .prompting import mock_label

SCAN_SLEEP = 0.2  # pause between real API calls; tests set this to 0


class ScanInProgressError(RuntimeError):
    pass


@dataclass
class ScanJob:
    session_id: int
    status: str = "running"  # running | done | cancelled | error
    total: int = 0
    done: int = 0
    errors: int = 0
    source: str = "mock"  # mock | ai
    model: str = "mock"
    error_detail: str = ""
    started_at: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)


_jobs: dict[int, ScanJob] = {}
_lock = threading.Lock()


def get_job(session_id: int) -> Optional[ScanJob]:
    with _lock:
        return _jobs.get(session_id)


def cancel_scan(session_id: int) -> bool:
    with _lock:
        job = _jobs.get(session_id)
    if job is None or job.status != "running":
        return False
    job.cancel.set()
    return True


def snapshot(job: ScanJob) -> dict:
    return {
        "status": job.status,
        "done": job.done,
        "total": job.total,
        "errors": job.errors,
        "source": job.source,
        "model": job.model,
        "error_detail": job.error_detail,
    }


def start_scan(
    session_id: int, mock: bool, model_override: Optional[str] = None
) -> ScanJob:
    with _lock:
        existing = _jobs.get(session_id)
        if existing is not None and existing.status == "running":
            raise ScanInProgressError("A scan is already running for this session.")

        with get_conn() as conn:
            total = repository.window_count(conn, session_id)
            config = None if mock else resolve_ai_config(conn, model_override)
            # A fresh scan replaces prior verdicts so stale flags can't linger.
            repository.delete_ai_suggestions(conn, session_id)

        job = ScanJob(
            session_id=session_id,
            total=total,
            source="ai" if config else "mock",
            model=config.model if config else "mock",
            started_at=int(time.time()),
        )
        _jobs[session_id] = job

    threading.Thread(target=_run, args=(job, config), daemon=True).start()
    return job


def _run(job: ScanJob, config: Optional[DeepSeekConfig]) -> None:
    labeler = DeepSeekLabeler(config) if config else None
    try:
        for seq in range(job.total):
            if job.cancel.is_set():
                job.status = "cancelled"
                return
            with get_conn() as conn:
                window = repository.get_window(conn, job.session_id, seq)
            if window is None:  # session deleted mid-scan
                job.status = "cancelled"
                job.error_detail = "Session no longer exists."
                return
            try:
                suggestion = mock_label(window) if labeler is None else labeler.label_window(window)
            except Exception as exc:  # noqa: BLE001 - keep scanning past bad windows.
                job.errors += 1
                job.error_detail = str(exc)
                continue
            # Fresh connection per window so each verdict commits independently.
            with get_conn() as conn:
                repository.upsert_ai_suggestion(
                    conn,
                    job.session_id,
                    window["window_pk"],
                    {**suggestion, "source": job.source, "model": job.model},
                )
            job.done += 1
            if labeler is not None and seq < job.total - 1 and SCAN_SLEEP > 0:
                time.sleep(SCAN_SLEEP)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - surface fatal errors to the UI.
        job.status = "error"
        job.error_detail = str(exc)
