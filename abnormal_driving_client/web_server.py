from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DEFAULT_SAMPLE_CSV = PROJECT_ROOT / "gps_sample.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend_runtime import BackendRuntime, to_jsonable


class DashboardHub:
    def __init__(self) -> None:
        self.queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.latest_state: dict[str, Any] = {}
        self.warning_events: list[dict[str, Any]] = []

    def bind_loop(self) -> None:
        self.loop = asyncio.get_running_loop()

    def emit(self, message: dict[str, Any]) -> None:
        event = to_jsonable(message)
        event_name = event.get("event")
        payload = event.get("payload", {})

        if event_name == "state_snapshot" and isinstance(payload, dict):
            self.latest_state = payload
        elif event_name == "warning_logged" and isinstance(payload, dict):
            self.warning_events = [payload, *self.warning_events[:49]]
        elif event_name == "warnings_reset":
            self.warning_events = []

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._broadcast, event)

    def _broadcast(self, event: dict[str, Any]) -> None:
        for queue in list(self.queues):
            queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.queues.discard(queue)

    def state_payload(self) -> dict[str, Any]:
        return {
            "state": self.latest_state,
            "warnings": self.warning_events,
        }


hub = DashboardHub()
runtime = BackendRuntime(hub.emit)
app = FastAPI(title="Abnormal Driving Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    hub.bind_loop()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    runtime.start()
    runtime.set_language("en")
    if DEFAULT_SAMPLE_CSV.exists():
        runtime.select_csv(str(DEFAULT_SAMPLE_CSV))


@app.on_event("shutdown")
async def shutdown() -> None:
    runtime.shutdown()


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    runtime.emit_state()
    return hub.state_payload()


@app.get("/api/events")
async def stream_events() -> StreamingResponse:
    queue = hub.subscribe()

    async def event_generator():
        yield format_sse({"event": "state_snapshot", "payload": hub.latest_state})
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield format_sse(event)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/csv/upload")
async def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    safe_name = Path(file.filename).name
    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    runtime.select_csv(str(target))
    return {"ok": True, "filename": safe_name, "state": hub.latest_state}


@app.post("/api/replay/start")
async def start_replay() -> dict[str, Any]:
    runtime.start_csv_simulation()
    return {"ok": True, "state": hub.latest_state}


@app.post("/api/replay/stop")
async def stop_replay() -> dict[str, Any]:
    runtime.stop_csv_simulation()
    runtime.emit_state()
    return {"ok": True, "state": hub.latest_state}


@app.post("/api/replay/reset")
async def reset_replay() -> dict[str, Any]:
    runtime.reset_replay_state()
    return {"ok": True, "state": hub.latest_state}


@app.post("/api/language")
async def set_language(payload: dict[str, str]) -> dict[str, Any]:
    runtime.set_language(payload.get("language", "en"))
    return {"ok": True, "state": hub.latest_state}


def format_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(to_jsonable(event), ensure_ascii=False)}\n\n"
