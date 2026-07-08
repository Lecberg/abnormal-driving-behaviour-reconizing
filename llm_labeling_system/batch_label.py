"""Batch-label GPS trajectory windows with DeepSeek (or a deterministic mock),
writing results into the SQLite store and exporting CSV/JSONL.

For thesis use, treat these LLM labels as weak labels; manually reviewed labels
from the web tool are stronger evidence.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import repository
from .config import settings
from .db import get_conn, init_db
from .services.deepseek_client import DeepSeekConfig, DeepSeekLabeler
from .services.export import export_csv, export_jsonl
from .services.prompting import mock_label
from .services.windowing import (
    DEFAULT_INPUT_CSV,
    WindowConfig,
    generate_windows,
    load_gps_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-label GPS trajectory windows into the SQLite store."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input CSV/XLSX path.")
    parser.add_argument("--window-size", type=int, default=10, help="Points per window.")
    parser.add_argument("--stride", type=int, default=5, help="Step between windows.")
    parser.add_argument("--limit", type=int, default=20, help="Max windows to label this run.")
    parser.add_argument("--max-rows", type=int, default=None, help="Max input rows to read.")
    parser.add_argument("--max-windows", type=int, default=500, help="Max windows to generate.")
    parser.add_argument("--model", default=None, help="DeepSeek model (default: config).")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock labels.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls.")
    parser.add_argument("--export-dir", default=str(settings.export_dir), help="Export directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    source_name = Path(args.input).name

    df = load_gps_data(args.input, max_rows=args.max_rows)
    windows = generate_windows(
        df,
        WindowConfig(window_size=args.window_size, stride=args.stride),
        max_windows=args.max_windows,
    )
    print(f"Loaded rows: {len(df)}  Generated windows: {len(windows)}")
    if not windows:
        print("No windows generated; nothing to label.")
        return

    labeler = None if args.mock else DeepSeekLabeler(DeepSeekConfig.from_env(model=args.model))

    with get_conn() as conn:
        project_id = repository.get_or_create_project(conn, "batch")
        session_id = repository.find_batch_session(
            conn, project_id, source_name, args.window_size, args.stride, args.max_windows
        )
        if session_id is None:
            session_id = repository.create_session(
                conn,
                source_name=source_name,
                source_path=str(args.input),
                window_size=args.window_size,
                stride=args.stride,
                max_rows=args.max_rows,
                max_windows=args.max_windows,
                row_count=int(len(df)),
                windows=windows,
                project_id=project_id,
            )
            print(f"Created batch session {session_id}")
        else:
            print(f"Resuming batch session {session_id}")
        pending = repository.iter_unlabeled_windows(conn, session_id)

    print(f"Unlabeled windows: {len(pending)}  Mode: {'mock' if args.mock else 'deepseek'}")

    model_name = "mock" if args.mock else (args.model or settings.deepseek_model)
    labeled = 0
    for window in pending:
        if labeled >= args.limit:
            break
        try:
            label = mock_label(window) if args.mock else labeler.label_window(window)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - keep going on individual failures.
            print(f"failed: {window['window_id']} {exc}")
            continue
        with get_conn() as conn:
            repository.upsert_label(
                conn,
                session_id,
                window["window_pk"],
                {**label, "source": "mock" if args.mock else "deepseek", "model": model_name},
            )
        labeled += 1
        print(f"{labeled}: {window['window_id']} {label['label']} conf={label['confidence']}")
        if not args.mock and args.sleep > 0:
            time.sleep(args.sleep)

    export_dir = Path(args.export_dir)
    csv_path = export_csv(export_dir / f"batch_{session_id}_labels.csv", session_id)
    jsonl_path = export_jsonl(export_dir / f"batch_{session_id}_labels.jsonl", session_id)
    print(f"Labeled {labeled} windows.\nExports:\n  {csv_path}\n  {jsonl_path}")


if __name__ == "__main__":
    main()
