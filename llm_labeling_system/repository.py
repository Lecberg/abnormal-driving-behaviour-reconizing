"""Data-access layer over SQLite. All functions take an open connection so the
caller controls the transaction (see db.get_conn)."""

from __future__ import annotations

import json
import time
from sqlite3 import Connection, Row
from typing import Any, Iterator, Optional


def _now() -> int:
    return int(time.time())


# --- projects -------------------------------------------------------------

def get_default_project_id(conn: Connection) -> int:
    return get_or_create_project(conn, "default")


def get_or_create_project(conn: Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO projects (name, created_at) VALUES (?,?)", (name, _now())
    )
    return int(cur.lastrowid)


# --- sessions -------------------------------------------------------------

def create_session(
    conn: Connection,
    *,
    source_name: str,
    source_path: Optional[str],
    window_size: int,
    stride: int,
    max_rows: Optional[int],
    max_windows: int,
    row_count: int,
    windows: list[dict[str, Any]],
    project_id: Optional[int] = None,
) -> int:
    if project_id is None:
        project_id = get_default_project_id(conn)
    cur = conn.execute(
        """INSERT INTO sessions
           (project_id, source_name, source_path, window_size, stride, max_rows,
            max_windows, row_count, window_count, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (project_id, source_name, source_path, window_size, stride, max_rows,
         max_windows, row_count, len(windows), _now()),
    )
    session_id = int(cur.lastrowid)
    conn.executemany(
        """INSERT INTO windows
           (session_id, seq, window_hash, vehicle_id, start_index, start_time,
            end_time, point_count, summary_json, rows_json)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                session_id, seq, w["window_id"], w["vehicle_id"], w["start_index"],
                w["start_time"], w["end_time"], w["point_count"],
                json.dumps(w["summary"], ensure_ascii=False),
                json.dumps(w["rows"], ensure_ascii=False),
            )
            for seq, w in enumerate(windows)
        ],
    )
    return session_id


def get_session_row(conn: Connection, session_id: int) -> Optional[Row]:
    return conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()


def get_session(conn: Connection, session_id: int) -> Optional[dict[str, Any]]:
    row = conn.execute(
        """SELECT s.*, p.name AS project_name FROM sessions s
           JOIN projects p ON p.id = s.project_id WHERE s.id=?""",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    return _session_summary(conn, row)


def list_sessions(conn: Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT s.*, p.name AS project_name FROM sessions s
           JOIN projects p ON p.id = s.project_id
           ORDER BY s.created_at DESC, s.id DESC"""
    ).fetchall()
    return [_session_summary(conn, r) for r in rows]


def delete_session(conn: Connection, session_id: int) -> Optional[str]:
    row = conn.execute(
        "SELECT source_path FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if not row:
        return None
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return row["source_path"]


def session_progress(conn: Connection, session_id: int) -> dict[str, int]:
    total = conn.execute(
        "SELECT COUNT(*) c FROM windows WHERE session_id=?", (session_id,)
    ).fetchone()["c"]
    labeled = conn.execute(
        "SELECT COUNT(*) c FROM labels WHERE session_id=?", (session_id,)
    ).fetchone()["c"]
    return {"total": total, "labeled": labeled, "remaining": max(0, total - labeled)}


def _session_summary(conn: Connection, row: Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project_name"],
        "source_name": row["source_name"],
        "row_count": row["row_count"],
        "window_size": row["window_size"],
        "stride": row["stride"],
        "window_count": row["window_count"],
        "created_at": row["created_at"],
        "progress": session_progress(conn, row["id"]),
    }


# --- windows --------------------------------------------------------------

def window_count(conn: Connection, session_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) c FROM windows WHERE session_id=?", (session_id,)
    ).fetchone()["c"]


def get_window(conn: Connection, session_id: int, seq: int) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM windows WHERE session_id=? AND seq=?", (session_id, seq)
    ).fetchone()
    return _window_payload(row) if row else None


def get_window_pk(conn: Connection, session_id: int, seq: int) -> Optional[int]:
    row = conn.execute(
        "SELECT id FROM windows WHERE session_id=? AND seq=?", (session_id, seq)
    ).fetchone()
    return row["id"] if row else None


def find_batch_session(
    conn: Connection,
    project_id: int,
    source_name: str,
    window_size: int,
    stride: int,
    max_windows: int,
) -> Optional[int]:
    row = conn.execute(
        """SELECT id FROM sessions
           WHERE project_id=? AND source_name=? AND window_size=? AND stride=? AND max_windows=?
           ORDER BY id DESC LIMIT 1""",
        (project_id, source_name, window_size, stride, max_windows),
    ).fetchone()
    return row["id"] if row else None


def iter_unlabeled_windows(conn: Connection, session_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT w.* FROM windows w
           LEFT JOIN labels l ON l.window_pk = w.id
           WHERE w.session_id=? AND l.id IS NULL
           ORDER BY w.seq""",
        (session_id,),
    ).fetchall()
    return [_window_payload(r) for r in rows]


def next_unlabeled_seq(conn: Connection, session_id: int) -> Optional[int]:
    row = conn.execute(
        """SELECT w.seq FROM windows w
           LEFT JOIN labels l ON l.window_pk = w.id
           WHERE w.session_id=? AND l.id IS NULL
           ORDER BY w.seq LIMIT 1""",
        (session_id,),
    ).fetchone()
    return row["seq"] if row else None


def _window_payload(row: Row) -> dict[str, Any]:
    return {
        "seq": row["seq"],
        "window_pk": row["id"],
        "window_id": row["window_hash"],
        "vehicle_id": row["vehicle_id"],
        "start_index": row["start_index"],
        "point_count": row["point_count"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "summary": json.loads(row["summary_json"] or "{}"),
        "rows": json.loads(row["rows_json"] or "[]"),
    }


# --- labels ---------------------------------------------------------------

def upsert_label(
    conn: Connection, session_id: int, window_pk: int, data: dict[str, Any]
) -> None:
    now = _now()
    conn.execute(
        """INSERT INTO labels
           (session_id, window_pk, label, confidence, risk_level, use_for_training,
            human_review_needed, reason, evidence_json, data_quality_flags_json,
            source, model, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(session_id, window_pk) DO UPDATE SET
             label=excluded.label,
             confidence=excluded.confidence,
             risk_level=excluded.risk_level,
             use_for_training=excluded.use_for_training,
             human_review_needed=excluded.human_review_needed,
             reason=excluded.reason,
             evidence_json=excluded.evidence_json,
             data_quality_flags_json=excluded.data_quality_flags_json,
             source=excluded.source,
             model=excluded.model,
             updated_at=excluded.updated_at""",
        (
            session_id, window_pk, data["label"], float(data["confidence"]),
            data.get("risk_level"), int(bool(data.get("use_for_training"))),
            int(bool(data.get("human_review_needed"))), data.get("reason", ""),
            json.dumps(data.get("evidence", []), ensure_ascii=False),
            json.dumps(data.get("data_quality_flags", []), ensure_ascii=False),
            data.get("source", "manual"), data.get("model", "human"), now, now,
        ),
    )


def get_label(conn: Connection, session_id: int, window_pk: int) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM labels WHERE session_id=? AND window_pk=?", (session_id, window_pk)
    ).fetchone()
    return _label_obj(row) if row else None


def _label_obj(row: Row) -> dict[str, Any]:
    return {
        "label": row["label"],
        "confidence": row["confidence"],
        "risk_level": row["risk_level"],
        "evidence": json.loads(row["evidence_json"] or "[]"),
        "reason": row["reason"] or "",
        "data_quality_flags": json.loads(row["data_quality_flags_json"] or "[]"),
        "use_for_training": bool(row["use_for_training"]),
        "human_review_needed": bool(row["human_review_needed"]),
        "source": row["source"],
        "model": row["model"],
    }


def list_labels(
    conn: Connection,
    session_id: int,
    label: str = "all",
    min_confidence: float = 0.0,
    training_only: bool = False,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT l.*, w.window_hash, w.vehicle_id, w.start_time, w.end_time, w.summary_json
           FROM labels l JOIN windows w ON w.id = l.window_pk
           WHERE l.session_id=? ORDER BY w.seq""",
        (session_id,),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for r in rows:
        if label != "all" and r["label"] != label:
            continue
        if float(r["confidence"] or 0) < min_confidence:
            continue
        if training_only and not r["use_for_training"]:
            continue
        summary = json.loads(r["summary_json"] or "{}")
        items.append({
            "window_id": r["window_hash"],
            "vehicle_id": r["vehicle_id"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "label": r["label"],
            "confidence": r["confidence"],
            "risk_level": r["risk_level"],
            "use_for_training": bool(r["use_for_training"]),
            "reason": r["reason"],
            "source": r["source"],
            "max_gps_speed": summary.get("max_gps_speed"),
            "total_heading_change": summary.get("total_heading_change"),
            "brake_count": summary.get("brake_count"),
            "quality": summary.get("data_quality_flags", []),
        })
    return items


def iter_label_export_rows(
    conn: Connection, session_id: Optional[int] = None
) -> Iterator[Row]:
    query = (
        """SELECT l.*, w.window_hash, w.vehicle_id, w.start_time, w.end_time,
                  s.source_name
           FROM labels l
           JOIN windows w ON w.id = l.window_pk
           JOIN sessions s ON s.id = l.session_id"""
    )
    params: tuple[Any, ...] = ()
    if session_id is not None:
        query += " WHERE l.session_id=?"
        params = (session_id,)
    query += " ORDER BY l.session_id, w.seq"
    yield from conn.execute(query, params)


# --- ai suggestions ---------------------------------------------------------

def upsert_ai_suggestion(
    conn: Connection, session_id: int, window_pk: int, data: dict[str, Any]
) -> None:
    now = _now()
    conn.execute(
        """INSERT INTO ai_suggestions
           (session_id, window_pk, label, confidence, risk_level, use_for_training,
            human_review_needed, reason, evidence_json, data_quality_flags_json,
            source, model, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(session_id, window_pk) DO UPDATE SET
             label=excluded.label,
             confidence=excluded.confidence,
             risk_level=excluded.risk_level,
             use_for_training=excluded.use_for_training,
             human_review_needed=excluded.human_review_needed,
             reason=excluded.reason,
             evidence_json=excluded.evidence_json,
             data_quality_flags_json=excluded.data_quality_flags_json,
             source=excluded.source,
             model=excluded.model,
             updated_at=excluded.updated_at""",
        (
            session_id, window_pk, data["label"], float(data["confidence"]),
            data.get("risk_level"), int(bool(data.get("use_for_training"))),
            int(bool(data.get("human_review_needed"))), data.get("reason", ""),
            json.dumps(data.get("evidence", []), ensure_ascii=False),
            json.dumps(data.get("data_quality_flags", []), ensure_ascii=False),
            data.get("source", "ai"), data.get("model", "unknown"), now, now,
        ),
    )


def list_flagged_suggestions(conn: Connection, session_id: int) -> list[dict[str, Any]]:
    """AI verdicts worth a human look: risky labels plus 'unclear' (can't judge).

    Not filtered on human_review_needed: the mock labeler sets it on every
    window (its confidence is always < 0.8), which would flag whole sessions.
    """
    rows = conn.execute(
        """SELECT a.*, w.seq FROM ai_suggestions a
           JOIN windows w ON w.id = a.window_pk
           WHERE a.session_id=? AND a.label != 'normal'
           ORDER BY w.seq""",
        (session_id,),
    ).fetchall()
    return [
        {
            "seq": row["seq"],
            "source": row["source"],
            "model": row["model"],
            "suggestion": _label_obj(row),
        }
        for row in rows
    ]


def iter_ai_suggestion_export_rows(
    conn: Connection, session_id: Optional[int] = None
) -> Iterator[Row]:
    query = (
        """SELECT a.*, w.window_hash, w.vehicle_id, w.start_time, w.end_time,
                  s.source_name
           FROM ai_suggestions a
           JOIN windows w ON w.id = a.window_pk
           JOIN sessions s ON s.id = a.session_id"""
    )
    params: tuple[Any, ...] = ()
    if session_id is not None:
        query += " WHERE a.session_id=?"
        params = (session_id,)
    query += " ORDER BY a.session_id, w.seq"
    yield from conn.execute(query, params)


def delete_ai_suggestions(conn: Connection, session_id: int) -> None:
    conn.execute("DELETE FROM ai_suggestions WHERE session_id=?", (session_id,))


def count_ai_suggestions(conn: Connection, session_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM ai_suggestions WHERE session_id=?", (session_id,)
    ).fetchone()
    return int(row["n"])


# --- app settings -----------------------------------------------------------

def get_setting(conn: Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(conn: Connection, key: str, value: Optional[str]) -> None:
    """Store a setting; empty/None deletes the row (absent row = fall back to env)."""
    value = (value or "").strip()
    if not value:
        conn.execute("DELETE FROM app_settings WHERE key=?", (key,))
        return
    conn.execute(
        """INSERT INTO app_settings (key, value, updated_at) VALUES (?,?,?)
           ON CONFLICT(key) DO UPDATE SET
             value=excluded.value,
             updated_at=excluded.updated_at""",
        (key, value, _now()),
    )
