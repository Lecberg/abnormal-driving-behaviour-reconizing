"""One-time importer: load legacy JSONL label files (from the old file-based
tool, e.g. outputs/manual_labels.jsonl) into the SQLite store so no labeling
work is lost after the refactor.

Usage:
    python -m llm_labeling_system.import_legacy [path/to/manual_labels.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import repository
from .db import get_conn, init_db

DEFAULT_LEGACY = Path(__file__).resolve().parent / "outputs" / "manual_labels.jsonl"


def read_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def import_file(path: str | Path) -> None:
    records = read_records(path)
    if not records:
        print(f"No records found in {path}")
        return

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec.get("source_file") or "legacy", []).append(rec)

    init_db()
    with get_conn() as conn:
        project_id = repository.get_or_create_project(conn, "imported")
        for source_name, recs in groups.items():
            seen: set[str] = set()
            windows: list[dict[str, Any]] = []
            for rec in recs:
                w = rec.get("window", {})
                wid = w.get("window_id")
                if not wid or wid in seen:
                    continue
                seen.add(wid)
                windows.append({
                    "window_id": wid,
                    "vehicle_id": w.get("vehicle_id", ""),
                    "start_index": w.get("start_index", 0),
                    "point_count": w.get("point_count", 0),
                    "start_time": w.get("start_time", ""),
                    "end_time": w.get("end_time", ""),
                    "summary": w.get("summary", {}),
                    "rows": w.get("rows", []),
                })
            if not windows:
                continue

            session_id = repository.create_session(
                conn,
                source_name=source_name,
                source_path=None,
                window_size=windows[0]["point_count"] or 0,
                stride=0,
                max_rows=None,
                max_windows=len(windows),
                row_count=0,
                windows=windows,
                project_id=project_id,
            )

            label_count = 0
            for rec in recs:
                wid = rec.get("window", {}).get("window_id")
                if not wid:
                    continue
                pk_row = conn.execute(
                    "SELECT id FROM windows WHERE session_id=? AND window_hash=?",
                    (session_id, wid),
                ).fetchone()
                if not pk_row:
                    continue
                label = rec.get("label", {})
                repository.upsert_label(conn, session_id, pk_row["id"], {
                    "label": label.get("label", "unclear"),
                    "confidence": label.get("confidence", 0.0),
                    "risk_level": label.get("risk_level", "unclear"),
                    "use_for_training": label.get("use_for_training", False),
                    "human_review_needed": label.get("human_review_needed", False),
                    "reason": label.get("reason", ""),
                    "evidence": label.get("evidence", []),
                    "data_quality_flags": label.get("data_quality_flags", []),
                    "source": rec.get("source", "manual"),
                    "model": rec.get("model", "human"),
                })
                label_count += 1
            print(f"Imported session {session_id}: {source_name} "
                  f"({len(windows)} windows, {label_count} labels)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy JSONL labels into SQLite.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_LEGACY),
                        help="Path to a legacy *_labels.jsonl file.")
    args = parser.parse_args()
    import_file(args.path)


if __name__ == "__main__":
    main()
