from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


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


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_completed_ids(jsonl_path: str | Path) -> set[str]:
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["window"]["window_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_label_records(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    records = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_label_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    # Write to a temp file first so a crash mid-write cannot truncate saved labels.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp_path, target)


def export_csv(jsonl_path: str | Path, csv_path: str | Path) -> None:
    records = read_label_records(jsonl_path)
    with Path(csv_path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for record in records:
            window = record.get("window", {})
            label = record.get("label", {})
            writer.writerow(
                {
                    "window_id": window.get("window_id", ""),
                    "vehicle_id": window.get("vehicle_id", ""),
                    "start_time": window.get("start_time", ""),
                    "end_time": window.get("end_time", ""),
                    "label": label.get("label", ""),
                    "confidence": label.get("confidence", ""),
                    "risk_level": label.get("risk_level", ""),
                    "use_for_training": label.get("use_for_training", ""),
                    "reason": label.get("reason", ""),
                    "evidence": " | ".join(label.get("evidence", [])),
                    "data_quality_flags": " | ".join(label.get("data_quality_flags", [])),
                    "human_review_needed": label.get("human_review_needed", ""),
                    "source": record.get("source", ""),
                    "source_file": record.get("source_file", ""),
                }
            )
