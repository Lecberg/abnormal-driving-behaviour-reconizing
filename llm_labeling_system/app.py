from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ensure_dirs
from .db import init_db
from .routers import labels, meta, scan, sessions, settings, suggest, windows

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    ensure_dirs()
    init_db()

    app = FastAPI(title="Manual Driving Labeling")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8010",
            "http://127.0.0.1:8010",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta.router)
    app.include_router(sessions.router)
    app.include_router(windows.router)
    app.include_router(labels.router)
    app.include_router(suggest.router)
    app.include_router(settings.router)
    app.include_router(scan.router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
