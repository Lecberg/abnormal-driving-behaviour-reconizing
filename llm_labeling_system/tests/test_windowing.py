import pandas as pd
import pytest

from llm_labeling_system.services.windowing import (
    WindowConfig,
    circular_diffs,
    generate_windows,
    haversine_km,
    load_gps_data,
)


def _write_csv(tmp_path, rows=12, vid="veh1"):
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
    path = tmp_path / "data.csv"
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_generate_windows_count_and_summary(tmp_path):
    df = load_gps_data(_write_csv(tmp_path))
    windows = generate_windows(df, WindowConfig(window_size=5, stride=5))
    # 12 rows, window 5, stride 5 -> starts at 0 and 5 -> 2 windows.
    assert len(windows) == 2

    first = windows[0]
    assert first["point_count"] == 5
    assert first["summary"]["max_gps_speed"] == 34  # speeds 30..34
    assert first["summary"]["avg_gps_speed"] == 32
    assert first["summary"]["brake_count"] >= 1
    assert first["vehicle_id"] == "veh1"
    # stable, non-empty window id
    assert isinstance(first["window_id"], str) and len(first["window_id"]) == 16


def test_max_windows_caps_output(tmp_path):
    df = load_gps_data(_write_csv(tmp_path, rows=30))
    windows = generate_windows(df, WindowConfig(window_size=5, stride=1), max_windows=3)
    assert len(windows) == 3


def test_missing_vid_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"gps时间": ["2024-01-01 08:00:00"], "gps速度": [10]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_gps_data(path)


def test_circular_diffs_wraps_around():
    # 10deg -> 350deg is a 20deg change the short way, not 340.
    assert circular_diffs(pd.Series([10.0, 350.0])) == [20.0]


def test_haversine_zero_and_known():
    assert haversine_km((39.0, 116.0), (39.0, 116.0)) == 0
    # ~0.001 deg of latitude is ~0.111 km
    assert haversine_km((39.0, 116.0), (39.001, 116.0)) == pytest.approx(0.111, abs=0.01)
