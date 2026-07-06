from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .deepseek_client import DEFAULT_MODEL, DeepSeekConfig, DeepSeekLabeler
from .prompting import mock_label
from .storage import append_jsonl, ensure_output_dir, export_csv, read_completed_ids
from .windowing import DEFAULT_INPUT_CSV, WindowConfig, generate_windows, load_gps_data


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label GPS trajectory windows with DeepSeek.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--window-size", type=int, default=10, help="Number of points per trajectory window.")
    parser.add_argument("--stride", type=int, default=5, help="Step size between windows.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum windows to label in this run.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum input CSV rows to read.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock labels instead of DeepSeek.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output and relabel windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    labels_path = output_dir / "llm_labels.jsonl"
    csv_path = output_dir / "llm_labels.csv"
    failed_path = output_dir / "failed_windows.jsonl"

    if args.overwrite:
        for path in (labels_path, csv_path, failed_path):
            if path.exists():
                path.unlink()

    config = WindowConfig(window_size=args.window_size, stride=args.stride)
    df = load_gps_data(args.input, max_rows=args.max_rows)
    windows = generate_windows(df, config)
    completed_ids = set() if args.overwrite else read_completed_ids(labels_path)

    labeler = None if args.mock else DeepSeekLabeler(DeepSeekConfig.from_env(model=args.model))
    labeled_count = 0

    print(f"Loaded rows: {len(df)}")
    print(f"Generated windows: {len(windows)}")
    print(f"Already labeled windows: {len(completed_ids)}")
    print(f"Mode: {'mock' if args.mock else 'deepseek'}")

    for window in windows:
        if args.limit is not None and labeled_count >= args.limit:
            break
        if window["window_id"] in completed_ids:
            continue

        try:
            label = mock_label(window) if args.mock else labeler.label_window(window)  # type: ignore[union-attr]
            record = {
                "source": "mock" if args.mock else "deepseek",
                "model": "mock" if args.mock else args.model,
                "created_at": int(time.time()),
                "window": window,
                "label": label,
            }
            append_jsonl(labels_path, record)
            labeled_count += 1
            print(
                f"{labeled_count}: {window['window_id']} "
                f"{label['label']} confidence={label['confidence']}"
            )
            if not args.mock and args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001 - keep the failed window for later review.
            append_jsonl(
                failed_path,
                {
                    "created_at": int(time.time()),
                    "window": window,
                    "error": str(exc),
                },
            )
            print(f"failed: {window['window_id']} {exc}")

    export_csv(labels_path, csv_path)
    print(f"Saved JSONL: {labels_path}")
    print(f"Saved CSV: {csv_path}")
    if failed_path.exists():
        print(f"Failed windows: {failed_path}")


if __name__ == "__main__":
    main()
