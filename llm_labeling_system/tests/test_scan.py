import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from llm_labeling_system.app import app
from llm_labeling_system.db import get_conn
from llm_labeling_system.services import scan_jobs

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_scan_state():
    yield
    with scan_jobs._lock:
        scan_jobs._jobs.clear()
    with get_conn() as conn:
        conn.execute("DELETE FROM app_settings")


def _csv_bytes(rows=12, vid="veh1"):
    records = []
    for i in range(rows):
        records.append({
            "vid_md5": vid,
            "gps时间": f"2024-01-01 08:00:{i:02d}",
            "Lng": 116.0 + i * 0.001,
            "Lat": 39.0 + i * 0.001,
            "gps速度": 30 + i,
            "与正北方向夹角": (i * 40) % 360,
            "制动信号": 1 if i % 3 == 0 else 0,
        })
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")


def _create_session():
    resp = client.post(
        "/api/sessions",
        files={"file": ("veh1.csv", _csv_bytes(), "text/csv")},
        data={"window_size": "5", "stride": "5", "max_windows": "50"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait(session_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/sessions/{session_id}/scan").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("scan did not finish in time")


def test_mock_scan_end_to_end():
    sid = _create_session()["id"]
    resp = client.post(f"/api/sessions/{sid}/scan", json={"mock": True})
    assert resp.status_code == 200

    body = _wait(sid)
    assert body["status"] == "done"
    assert body["done"] == body["total"] == 2
    assert body["errors"] == 0
    assert body["source"] == "mock"
    for item in body["results"]:
        assert isinstance(item["seq"], int)
        s = item["suggestion"]
        assert s["label"] != "normal"
        for key in ("label", "confidence", "risk_level", "reason", "evidence",
                    "data_quality_flags", "use_for_training", "human_review_needed"):
            assert key in s


def test_double_start_conflict(monkeypatch):
    sid = _create_session()["id"]
    release = threading.Event()
    original = scan_jobs.mock_label

    def blocking_mock(window):
        release.wait(timeout=5)
        return original(window)

    monkeypatch.setattr(scan_jobs, "mock_label", blocking_mock)
    assert client.post(f"/api/sessions/{sid}/scan", json={"mock": True}).status_code == 200
    assert client.post(f"/api/sessions/{sid}/scan", json={"mock": True}).status_code == 409
    release.set()
    assert _wait(sid)["status"] == "done"


def test_cancel(monkeypatch):
    sid = _create_session()["id"]
    release = threading.Event()
    original = scan_jobs.mock_label

    def blocking_mock(window):
        release.wait(timeout=5)
        return original(window)

    monkeypatch.setattr(scan_jobs, "mock_label", blocking_mock)
    client.post(f"/api/sessions/{sid}/scan", json={"mock": True})
    resp = client.delete(f"/api/sessions/{sid}/scan")
    assert resp.status_code == 200
    release.set()
    body = _wait(sid)
    assert body["status"] == "cancelled"
    assert body["done"] < body["total"]


def test_ai_path_uses_configured_model(monkeypatch):
    sid = _create_session()["id"]
    client.put("/api/settings", json={"api_key": "sk-x", "model": "custom-scan-model"})
    monkeypatch.setattr(scan_jobs, "SCAN_SLEEP", 0)

    fixture = {
        "label": "speeding", "confidence": 0.9, "risk_level": "high",
        "use_for_training": True, "human_review_needed": False,
        "reason": "way too fast", "evidence": ["max speed"], "data_quality_flags": [],
    }
    monkeypatch.setattr(
        scan_jobs.DeepSeekLabeler, "label_window", lambda self, window: dict(fixture)
    )

    client.post(f"/api/sessions/{sid}/scan", json={"mock": False})
    body = _wait(sid)
    assert body["status"] == "done"
    assert body["source"] == "ai"
    assert body["model"] == "custom-scan-model"
    assert len(body["results"]) == 2
    assert all(r["model"] == "custom-scan-model" for r in body["results"])


def test_per_window_error_continues(monkeypatch):
    sid = _create_session()["id"]
    original = scan_jobs.mock_label
    calls = {"n": 0}

    def flaky_mock(window):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(window)

    monkeypatch.setattr(scan_jobs, "mock_label", flaky_mock)
    client.post(f"/api/sessions/{sid}/scan", json={"mock": True})
    body = _wait(sid)
    assert body["status"] == "done"
    assert body["errors"] == 1
    assert body["done"] == body["total"] - 1


def test_rescan_replaces_results():
    sid = _create_session()["id"]
    for _ in range(2):
        client.post(f"/api/sessions/{sid}/scan", json={"mock": True})
        body = _wait(sid)
        assert body["status"] == "done"
    with get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM ai_suggestions WHERE session_id=?", (sid,)
        ).fetchone()["n"]
    assert n == body["total"]


def test_results_survive_job_loss():
    sid = _create_session()["id"]
    client.post(f"/api/sessions/{sid}/scan", json={"mock": True})
    done = _wait(sid)
    assert done["status"] == "done"

    with scan_jobs._lock:
        scan_jobs._jobs.clear()  # simulate a server restart

    body = client.get(f"/api/sessions/{sid}/scan").json()
    assert body["status"] == "idle"
    assert body["results"] == done["results"]


def test_export_ai_scan_results():
    sid = _create_session()["id"]
    client.post(f"/api/sessions/{sid}/scan", json={"mock": True})
    assert _wait(sid)["status"] == "done"

    resp = client.get(f"/api/sessions/{sid}/export?source=ai")
    assert resp.status_code == 200
    text = resp.content.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0].split(",")[:5] == ["window_id", "vehicle_id", "start_time", "end_time", "label"]
    assert len(lines) == 3  # header + 2 scanned windows
    assert "mock" in text

    # Human-label export is unaffected (no labels saved yet -> header only).
    manual = client.get(f"/api/sessions/{sid}/export").content.decode("utf-8-sig")
    assert len(manual.strip().splitlines()) == 1

    assert client.get(f"/api/sessions/{sid}/export?source=bogus").status_code == 400


def test_scan_unknown_session_404():
    assert client.post("/api/sessions/99999/scan", json={"mock": True}).status_code == 404
    assert client.get("/api/sessions/99999/scan").status_code == 404
