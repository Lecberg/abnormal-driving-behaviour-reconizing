"""Export DB label rows to CSV/JSONL, preserving the original column contract
so downstream thesis analysis keeps working."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from .. import repository
from ..db import get_conn


def _export_rows(conn, table: str, session_id: Optional[int]):
    if table == "ai":
        return repository.iter_ai_suggestion_export_rows(conn, session_id)
    return repository.iter_label_export_rows(conn, session_id)


OUTPUT_FIELDS = [
    "window_id",
    "vehicle_id",
    "start_time",
    "end_time",
    "label",
    "confidence",
    "risk_level",
    "use_for_training",
    "reason",
    "evidence",
    "data_quality_flags",
    "human_review_needed",
    "source",
    "source_file",
]


def export_csv(
    csv_path: str | Path, session_id: Optional[int] = None, table: str = "labels"
) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn, path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in _export_rows(conn, table, session_id):
            writer.writerow({
                "window_id": r["window_hash"],
                "vehicle_id": r["vehicle_id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "label": r["label"],
                "confidence": r["confidence"],
                "risk_level": r["risk_level"],
                "use_for_training": bool(r["use_for_training"]),
                "reason": r["reason"] or "",
                "evidence": " | ".join(json.loads(r["evidence_json"] or "[]")),
                "data_quality_flags": " | ".join(json.loads(r["data_quality_flags_json"] or "[]")),
                "human_review_needed": bool(r["human_review_needed"]),
                "source": r["source"],
                "source_file": r["source_name"],
            })
    return path


def export_jsonl(
    jsonl_path: str | Path, session_id: Optional[int] = None, table: str = "labels"
) -> Path:
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn, path.open("w", encoding="utf-8", newline="") as file:
        for r in _export_rows(conn, table, session_id):
            record = {
                "source": r["source"],
                "source_file": r["source_name"],
                "model": r["model"],
                "created_at": r["created_at"],
                "window": {
                    "window_id": r["window_hash"],
                    "vehicle_id": r["vehicle_id"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                },
                "label": {
                    "label": r["label"],
                    "confidence": r["confidence"],
                    "risk_level": r["risk_level"],
                    "evidence": json.loads(r["evidence_json"] or "[]"),
                    "reason": r["reason"] or "",
                    "data_quality_flags": json.loads(r["data_quality_flags_json"] or "[]"),
                    "use_for_training": bool(r["use_for_training"]),
                    "human_review_needed": bool(r["human_review_needed"]),
                },
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
