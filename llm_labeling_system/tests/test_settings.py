import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from llm_labeling_system.app import app
from llm_labeling_system.db import get_conn
from llm_labeling_system.services import ai_settings
from llm_labeling_system.services.deepseek_client import (
    DeepSeekConfig,
    DeepSeekLabeler,
    test_connection as check_connection,  # aliased so pytest doesn't collect it
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_settings():
    yield
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


def test_settings_roundtrip_never_echoes_key():
    resp = client.put("/api/settings", json={
        "base_url": "https://api.example.com/v1",
        "model": "my-model",
        "api_key": "sk-secret-123",
    })
    assert resp.status_code == 200
    assert "sk-secret-123" not in resp.text
    body = resp.json()
    assert body["has_api_key"] is True
    assert body["api_key_source"] == "db"
    assert body["base_url"] == "https://api.example.com/v1"
    assert body["model"] == "my-model"

    got = client.get("/api/settings")
    assert "sk-secret-123" not in got.text
    assert got.json()["has_api_key"] is True


def test_partial_update_preserves_key():
    client.put("/api/settings", json={"api_key": "sk-keep", "base_url": "https://a.example"})
    body = client.put("/api/settings", json={"model": "gpt-4o-mini"}).json()
    assert body["has_api_key"] is True
    assert body["base_url"] == "https://a.example"
    assert body["model"] == "gpt-4o-mini"


def test_clear_key_and_config_flag():
    client.put("/api/settings", json={"api_key": "sk-temp"})
    assert client.get("/api/config").json()["deepseek_available"] is True

    body = client.put("/api/settings", json={"api_key": ""}).json()
    assert body["has_api_key"] is False
    assert body["api_key_source"] is None
    assert client.get("/api/config").json()["deepseek_available"] is False


def test_empty_values_revert_to_defaults():
    client.put("/api/settings", json={"base_url": "https://a.example", "model": "m1"})
    body = client.put("/api/settings", json={"base_url": "", "model": ""}).json()
    assert body["base_url"] == "https://api.deepseek.com"
    assert body["model"] == "deepseek-v4-flash"


def test_base_url_validation():
    resp = client.put("/api/settings", json={"base_url": "ftp://nope"})
    assert resp.status_code == 400


def test_env_vs_db_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert client.get("/api/settings").json()["api_key_source"] == "env"

    client.put("/api/settings", json={"api_key": "db-key"})
    assert client.get("/api/settings").json()["api_key_source"] == "db"
    with get_conn() as conn:
        config = ai_settings.resolve_ai_config(conn)
    assert config.api_key == "db-key"


def test_suggest_uses_db_config(monkeypatch):
    session = _create_session()
    sid = session["id"]
    client.put("/api/settings", json={"api_key": "sk-x", "model": "custom-model"})

    fixture = {
        "label": "normal", "confidence": 0.9, "risk_level": "low",
        "use_for_training": True, "human_review_needed": False,
        "reason": "ok", "evidence": [], "data_quality_flags": [],
    }
    monkeypatch.setattr(DeepSeekLabeler, "label_window", lambda self, window: fixture)

    body = client.post(f"/api/sessions/{sid}/suggest", json={"seq": 0, "mock": False}).json()
    assert body["source"] == "ai"
    assert body["model"] == "custom-model"


def test_suggest_falls_back_to_mock_without_key():
    session = _create_session()
    sid = session["id"]
    body = client.post(f"/api/sessions/{sid}/suggest", json={"seq": 0, "mock": False}).json()
    assert body["source"] == "mock"


class _StubResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_test_endpoint_ok_and_failure(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _StubResponse(200))
    body = client.post("/api/settings/test", json={"api_key": "sk-x", "model": "m"}).json()
    assert body["ok"] is True
    assert body["model"] == "m"

    monkeypatch.setattr(
        requests, "post",
        lambda *a, **kw: _StubResponse(401, {"error": {"message": "bad key"}}),
    )
    body = client.post("/api/settings/test", json={"api_key": "sk-x"}).json()
    assert body["ok"] is False
    assert "bad key" in body["detail"]

    def _boom(*a, **kw):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", _boom)
    body = client.post("/api/settings/test", json={"api_key": "sk-x"}).json()
    assert body["ok"] is False
    assert "refused" in body["detail"]


def test_test_endpoint_requires_some_key():
    resp = client.post("/api/settings/test", json={})
    assert resp.status_code == 400


def test_thinking_field_only_for_deepseek(monkeypatch):
    captured = {}

    def _capture(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _StubResponse(200)

    monkeypatch.setattr(requests, "post", _capture)

    check_connection(DeepSeekConfig(api_key="k", base_url="https://api.deepseek.com", model="m"))
    assert captured["body"].get("thinking") == {"type": "disabled"}

    check_connection(DeepSeekConfig(api_key="k", base_url="http://localhost:11434/v1", model="m"))
    assert "thinking" not in captured["body"]
