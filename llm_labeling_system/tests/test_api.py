import pandas as pd
import pytest
from fastapi.testclient import TestClient

from llm_labeling_system.app import app

client = TestClient(app)


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


def test_config_reports_default_model_and_no_key():
    body = client.get("/api/config").json()
    assert body["default_model"] == "deepseek-v4-flash"
    assert body["deepseek_available"] is False  # forced off in conftest
    assert "normal" in body["labels"]


def test_full_round_trip():
    session = _create_session()
    sid = session["id"]
    assert session["window_count"] == 2

    # next-unlabeled
    nxt = client.get(f"/api/sessions/{sid}/next-unlabeled").json()
    seq = nxt["window"]["seq"]

    # save
    saved = client.post(f"/api/sessions/{sid}/labels",
                        json={"seq": seq, "label": "speeding", "confidence": 0.8})
    assert saved.status_code == 200
    assert saved.json()["progress"]["labeled"] == 1

    # relabel same window -> still 1 labeled
    relabel = client.post(f"/api/sessions/{sid}/labels",
                          json={"seq": seq, "label": "normal", "confidence": 0.9})
    assert relabel.json()["progress"]["labeled"] == 1

    # export CSV keeps the original column header
    exp = client.get(f"/api/sessions/{sid}/export")
    assert exp.status_code == 200
    header = exp.content.decode("utf-8-sig").splitlines()[0]
    assert header.split(",")[:5] == ["window_id", "vehicle_id", "start_time", "end_time", "label"]


def test_suggest_mock_without_key():
    session = _create_session()
    sid = session["id"]
    resp = client.post(f"/api/sessions/{sid}/suggest", json={"seq": 0, "mock": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "mock"
    assert body["suggestion"]["label"] in {
        "normal", "speeding", "harsh_accel_brake", "zigzag_unstable", "unclear"
    }


def test_invalid_label_rejected():
    session = _create_session()
    resp = client.post(f"/api/sessions/{session['id']}/labels",
                       json={"seq": 0, "label": "not_a_label", "confidence": 1.0})
    assert resp.status_code == 400


def test_delete_session_cleans_up():
    session = _create_session()
    sid = session["id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
