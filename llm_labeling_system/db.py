from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import ensure_dirs, settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_name   TEXT NOT NULL,
    source_path   TEXT,
    window_size   INTEGER NOT NULL,
    stride        INTEGER NOT NULL,
    max_rows      INTEGER,
    max_windows   INTEGER NOT NULL,
    row_count     INTEGER NOT NULL,
    window_count  INTEGER NOT NULL,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    window_hash  TEXT NOT NULL,
    vehicle_id   TEXT,
    start_index  INTEGER,
    start_time   TEXT,
    end_time     TEXT,
    point_count  INTEGER,
    summary_json TEXT,
    rows_json    TEXT,
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_windows_session ON windows(session_id, seq);

CREATE TABLE IF NOT EXISTS labels (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    window_pk               INTEGER NOT NULL REFERENCES windows(id) ON DELETE CASCADE,
    label                   TEXT NOT NULL,
    confidence              REAL NOT NULL,
    risk_level              TEXT,
    use_for_training        INTEGER NOT NULL DEFAULT 0,
    human_review_needed     INTEGER NOT NULL DEFAULT 0,
    reason                  TEXT,
    evidence_json           TEXT,
    data_quality_flags_json TEXT,
    source                  TEXT,
    model                   TEXT,
    created_at              INTEGER NOT NULL,
    updated_at              INTEGER NOT NULL,
    UNIQUE(session_id, window_pk)
);
CREATE INDEX IF NOT EXISTS idx_labels_session ON labels(session_id);

CREATE TABLE IF NOT EXISTS ai_suggestions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    window_pk               INTEGER NOT NULL REFERENCES windows(id) ON DELETE CASCADE,
    label                   TEXT NOT NULL,
    confidence              REAL NOT NULL,
    risk_level              TEXT,
    use_for_training        INTEGER NOT NULL DEFAULT 0,
    human_review_needed     INTEGER NOT NULL DEFAULT 0,
    reason                  TEXT,
    evidence_json           TEXT,
    data_quality_flags_json TEXT,
    source                  TEXT,
    model                   TEXT,
    created_at              INTEGER NOT NULL,
    updated_at              INTEGER NOT NULL,
    UNIQUE(session_id, window_pk)
);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_session ON ai_suggestions(session_id);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(settings.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Transaction-scoped connection: commit on success, rollback on error."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO projects (name, created_at) "
            "VALUES ('default', strftime('%s','now'))"
        )
