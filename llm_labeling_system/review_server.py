from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .deepseek_client import DEFAULT_MODEL, DeepSeekConfig, DeepSeekLabeler
from .prompting import mock_label
from .storage import export_csv, read_label_records, write_label_records
from .windowing import WindowConfig, generate_windows, load_gps_data


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

LLM_LABELS_JSONL = OUTPUT_DIR / "llm_labels.jsonl"
LLM_LABELS_CSV = OUTPUT_DIR / "llm_labels.csv"
MANUAL_SESSION_JSON = OUTPUT_DIR / "manual_session.json"
MANUAL_LABELS_JSONL = OUTPUT_DIR / "manual_labels.jsonl"
MANUAL_LABELS_CSV = OUTPUT_DIR / "manual_labels.csv"

ALLOWED_LABELS = {"normal", "speeding", "harsh_accel_brake", "zigzag_unstable", "unclear"}

app = FastAPI(title="Manual Driving Labeling")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Sync handler on purpose: pandas parsing blocks, and FastAPI moves sync handlers off the event loop.
@app.post("/api/manual/upload")
def upload_manual_file(
    file: UploadFile = File(...),
    window_size: int = Form(default=10),
    stride: int = Form(default=5),
    max_rows: int | None = Form(default=None),
    max_windows: int = Form(default=500),
) -> dict[str, Any]:
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        df = load_gps_data(target, max_rows=max_rows)
        windows = generate_windows(
            df,
            WindowConfig(window_size=window_size, stride=stride),
            max_windows=max_windows,
        )
    except Exception as exc:  # noqa: BLE001 - return a clear browser error.
        raise HTTPException(status_code=400, detail=f"Could not read this file: {exc}") from exc

    session = {
        "source_file": str(target),
        "source_name": Path(file.filename).name,
        "created_at": int(time.time()),
        "window_size": window_size,
        "stride": stride,
        "max_rows": max_rows,
        "max_windows": max_windows,
        "row_count": int(len(df)),
        "window_count": len(windows),
        "windows": windows,
    }
    write_json(MANUAL_SESSION_JSON, session)

    return manual_session_payload()


@app.get("/api/manual/session")
async def manual_session() -> dict[str, Any]:
    return manual_session_payload()


@app.get("/api/manual/window")
async def manual_window(index: int = Query(default=0)) -> dict[str, Any]:
    session = read_session_or_404()
    windows = session["windows"]
    if not windows:
        raise HTTPException(status_code=404, detail="No windows are available in this session.")
    if index < 0 or index >= len(windows):
        raise HTTPException(status_code=404, detail="Window index is out of range.")
    labels = labels_by_window_id()
    window = windows[index]
    return {
        "index": index,
        "total": len(windows),
        "window": window,
        "label": labels.get(window["window_id"]),
        "progress": progress_payload(session, labels),
    }


@app.get("/api/manual/next")
async def manual_next() -> dict[str, Any]:
    session = read_session_or_404()
    labels = labels_by_window_id()
    for index, window in enumerate(session["windows"]):
        if window["window_id"] not in labels:
            return {
                "index": index,
                "total": len(session["windows"]),
                "window": window,
                "label": None,
                "progress": progress_payload(session, labels),
            }
    return {"done": True, "progress": progress_payload(session, labels)}


# Sync handler on purpose: the DeepSeek HTTP call blocks for up to minutes and
# must not stall the event loop. FastAPI runs sync handlers in a thread pool.
@app.post("/api/manual/suggest")
def suggest_manual_label(payload: dict[str, Any]) -> dict[str, Any]:
    session = read_session_or_404()
    window_id = str(payload.get("window_id", "")).strip()
    model = str(payload.get("model") or DEFAULT_MODEL).strip()
    use_mock = bool(payload.get("mock", False))

    window_lookup = {window["window_id"]: window for window in session["windows"]}
    if window_id not in window_lookup:
        raise HTTPException(status_code=404, detail="Window not found in the active session.")

    window = window_lookup[window_id]
    if use_mock:
        suggestion = mock_label(window)
        source = "mock"
        model_name = "mock"
    else:
        try:
            config = DeepSeekConfig.from_env(model=model)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=400,
                detail="AI suggestion needs DEEPSEEK_API_KEY. Set it before starting the server.",
            ) from exc
        try:
            labeler = DeepSeekLabeler(config)
            suggestion = labeler.label_window(window)
            source = "deepseek"
            model_name = model
        except Exception as exc:  # noqa: BLE001 - show a clear browser error.
            raise HTTPException(status_code=502, detail=f"AI suggestion failed: {exc}") from exc

    return {
        "ok": True,
        "source": source,
        "model": model_name,
        "window_id": window_id,
        "suggestion": suggestion,
    }


@app.post("/api/manual/labels")
async def save_manual_label(payload: dict[str, Any]) -> dict[str, Any]:
    session = read_session_or_404()
    window_id = str(payload.get("window_id", "")).strip()
    label = str(payload.get("label", "")).strip()
    if label not in ALLOWED_LABELS:
        raise HTTPException(status_code=400, detail="Invalid label.")

    window_lookup = {window["window_id"]: window for window in session["windows"]}
    if window_id not in window_lookup:
        raise HTTPException(status_code=404, detail="Window not found in the active session.")

    confidence = clamp_float(payload.get("confidence", 1.0), 0.0, 1.0)
    notes = str(payload.get("notes", "")).strip()
    use_for_training = bool(payload.get("use_for_training", label != "unclear" and confidence >= 0.55))
    if label == "unclear" or confidence < 0.55:
        use_for_training = False

    manual_record = {
        "source": "manual",
        "source_file": session.get("source_name", ""),
        "model": "human",
        "created_at": int(time.time()),
        "window": window_lookup[window_id],
        "label": {
            "label": label,
            "confidence": confidence,
            "risk_level": risk_for_label(label),
            "evidence": ["manual_label"],
            "reason": notes,
            "data_quality_flags": window_lookup[window_id].get("summary", {}).get("data_quality_flags", []),
            "use_for_training": use_for_training,
            "human_review_needed": False,
        },
    }

    records = read_label_records(MANUAL_LABELS_JSONL)
    records = [record for record in records if record.get("window", {}).get("window_id") != window_id]
    records.append(manual_record)
    write_label_records(MANUAL_LABELS_JSONL, records)
    export_csv(MANUAL_LABELS_JSONL, MANUAL_LABELS_CSV)

    labels = labels_by_window_id()
    return {"ok": True, "record": manual_record, "progress": progress_payload(session, labels)}


@app.get("/api/manual/export")
async def export_manual() -> FileResponse:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_csv(MANUAL_LABELS_JSONL, MANUAL_LABELS_CSV)
    return FileResponse(MANUAL_LABELS_CSV, filename="manual_labels.csv")


@app.get("/api/labels")
async def labels(
    label: str = Query(default="all"),
    min_confidence: float = Query(default=0.0),
    training_only: bool = Query(default=False),
) -> dict[str, Any]:
    records = read_label_records(MANUAL_LABELS_JSONL)
    if not records:
        records = read_label_records(LLM_LABELS_JSONL)
    filtered = []
    for record in records:
        label_payload = record.get("label", {})
        if label != "all" and label_payload.get("label") != label:
            continue
        if float(label_payload.get("confidence", 0) or 0) < min_confidence:
            continue
        if training_only and not label_payload.get("use_for_training"):
            continue
        filtered.append(summary_record(record))
    return {"count": len(filtered), "items": filtered}


@app.get("/api/labels/{window_id}")
async def label_detail(window_id: str) -> dict[str, Any]:
    for source_path in (MANUAL_LABELS_JSONL, LLM_LABELS_JSONL):
        for record in read_label_records(source_path):
            if record.get("window", {}).get("window_id") == window_id:
                return record
    return {"error": "Window not found"}


@app.get("/api/export")
async def export() -> FileResponse:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANUAL_LABELS_JSONL.exists():
        export_csv(MANUAL_LABELS_JSONL, MANUAL_LABELS_CSV)
        return FileResponse(MANUAL_LABELS_CSV, filename="manual_labels.csv")
    export_csv(LLM_LABELS_JSONL, LLM_LABELS_CSV)
    return FileResponse(LLM_LABELS_CSV, filename="llm_labels.csv")


def manual_session_payload() -> dict[str, Any]:
    if not MANUAL_SESSION_JSON.exists():
        return {"has_session": False, "progress": {"total": 0, "labeled": 0, "remaining": 0}}
    session = read_json(MANUAL_SESSION_JSON)
    labels = labels_by_window_id()
    return {
        "has_session": True,
        "source_name": session.get("source_name", ""),
        "row_count": session.get("row_count", 0),
        "window_size": session.get("window_size", 0),
        "stride": session.get("stride", 0),
        "progress": progress_payload(session, labels),
    }


def progress_payload(session: dict[str, Any], labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    total = len(session.get("windows", []))
    labeled = sum(1 for window in session.get("windows", []) if window.get("window_id") in labels)
    return {"total": total, "labeled": labeled, "remaining": max(0, total - labeled)}


def labels_by_window_id() -> dict[str, dict[str, Any]]:
    return {
        record.get("window", {}).get("window_id"): record
        for record in read_label_records(MANUAL_LABELS_JSONL)
        if record.get("window", {}).get("window_id")
    }


def read_session_or_404() -> dict[str, Any]:
    if not MANUAL_SESSION_JSON.exists():
        raise HTTPException(status_code=404, detail="Upload a CSV or XLSX file first.")
    return read_json(MANUAL_SESSION_JSON)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def risk_for_label(label: str) -> str:
    if label == "normal":
        return "low"
    if label == "unclear":
        return "unclear"
    return "medium"


def summary_record(record: dict[str, Any]) -> dict[str, Any]:
    window = record.get("window", {})
    label = record.get("label", {})
    summary = window.get("summary", {})
    return {
        "window_id": window.get("window_id"),
        "vehicle_id": window.get("vehicle_id"),
        "start_time": window.get("start_time"),
        "end_time": window.get("end_time"),
        "label": label.get("label"),
        "confidence": label.get("confidence"),
        "risk_level": label.get("risk_level"),
        "use_for_training": label.get("use_for_training"),
        "reason": label.get("reason"),
        "source": record.get("source"),
        "max_gps_speed": summary.get("max_gps_speed"),
        "total_heading_change": summary.get("total_heading_change"),
        "brake_count": summary.get("brake_count"),
        "quality": summary.get("data_quality_flags", []),
    }
