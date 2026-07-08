from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# services/windowing.py -> parents[2] is the repository root (…/project_code).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "gps_1101.csv"

TIME_COLUMNS = ["gps时间", "gps鏃堕棿", "标准时间", "標準時間", "系统时间"]
NUMERIC_COLUMNS = [
    "Lng",
    "Lat",
    "gps速度",
    "vss速度",
    "海拔",
    "与正北方向夹角",
    "制动信号",
    "左转向灯信号",
    "右转向灯信号",
    "空档信号",
    "喇叭信号",
    "倒挡信号",
    "远光灯信号",
    "近光灯信号",
]


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 10
    stride: int = 5


def load_gps_data(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    suffix = csv_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(csv_path, nrows=max_rows, encoding="utf-8-sig", low_memory=False)
    elif suffix == ".xlsx":
        df = pd.read_excel(csv_path, nrows=max_rows)
    else:
        raise ValueError("Input file must be a CSV or XLSX file.")
    if "vid_md5" not in df.columns:
        raise ValueError("Input file must contain a vid_md5 column.")

    time_column = first_existing_column(df, TIME_COLUMNS)
    if not time_column:
        raise ValueError(f"Input file must contain one time column: {', '.join(TIME_COLUMNS)}")

    df = df.copy()
    df["_label_time"] = pd.to_datetime(df[time_column], format="mixed", errors="coerce")
    df = df.dropna(subset=["_label_time"])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["vid_md5", "_label_time"]).reset_index(drop=True)
    return df


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def generate_windows(
    df: pd.DataFrame, config: WindowConfig, max_windows: int | None = None
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for vehicle_id, vehicle_df in df.groupby("vid_md5", sort=False):
        vehicle_df = vehicle_df.sort_values("_label_time").reset_index(drop=True)
        if len(vehicle_df) < config.window_size:
            continue

        for start in range(0, len(vehicle_df) - config.window_size + 1, config.stride):
            if max_windows is not None and len(windows) >= max_windows:
                return windows
            window_df = vehicle_df.iloc[start : start + config.window_size].copy()
            windows.append(build_window_payload(str(vehicle_id), start, window_df))
    return windows


def build_window_payload(vehicle_id: str, start_index: int, window_df: pd.DataFrame) -> dict[str, Any]:
    start_time = format_time(window_df["_label_time"].iloc[0])
    end_time = format_time(window_df["_label_time"].iloc[-1])
    window_id = stable_window_id(vehicle_id, start_time, start_index, len(window_df))

    row_records = []
    for _, row in window_df.iterrows():
        row_records.append(
            {
                "time": format_time(row.get("_label_time")),
                "lng": clean_number(row.get("Lng")),
                "lat": clean_number(row.get("Lat")),
                "gps_speed": clean_number(row.get("gps速度")),
                "vss_speed": clean_number(row.get("vss速度")),
                "heading": clean_number(row.get("与正北方向夹角")),
                "brake_signal": clean_int(row.get("制动信号")),
                "left_turn_signal": clean_int(row.get("左转向灯信号")),
                "right_turn_signal": clean_int(row.get("右转向灯信号")),
                "neutral_signal": clean_int(row.get("空档信号")),
                "horn_signal": clean_int(row.get("喇叭信号")),
                "reverse_signal": clean_int(row.get("倒挡信号")),
                "road_id": clean_text(row.get("道路ID序列")),
                "vehicle_state": clean_text(row.get("车辆状态")),
                "acc_state": clean_text(row.get("ACC状态")),
                "district": clean_text(row.get("区县名称")),
            }
        )

    summary = summarize_window(window_df)
    return {
        "window_id": window_id,
        "vehicle_id": vehicle_id,
        "start_index": start_index,
        "point_count": int(len(window_df)),
        "start_time": start_time,
        "end_time": end_time,
        "summary": summary,
        "rows": row_records,
    }


def summarize_window(window_df: pd.DataFrame) -> dict[str, Any]:
    speeds = numeric_series(window_df, "gps速度")
    vss_speeds = numeric_series(window_df, "vss速度")
    headings = numeric_series(window_df, "与正北方向夹角")

    time_span_seconds = 0.0
    if len(window_df) > 1:
        delta = window_df["_label_time"].iloc[-1] - window_df["_label_time"].iloc[0]
        time_span_seconds = float(delta.total_seconds())

    heading_changes = circular_diffs(headings)
    gps_distance_km = total_distance_km(window_df)

    road_values = text_values(window_df, "道路ID序列")
    vehicle_states = sorted(set(text_values(window_df, "车辆状态")))
    districts = sorted(set(text_values(window_df, "区县名称")))

    return {
        "time_span_seconds": round(time_span_seconds, 3),
        "distance_km": round(gps_distance_km, 5) if gps_distance_km is not None else None,
        "avg_gps_speed": round_float(speeds.mean()),
        "max_gps_speed": round_float(speeds.max()),
        "min_gps_speed": round_float(speeds.min()),
        "gps_speed_delta": round_float(speeds.iloc[-1] - speeds.iloc[0]) if len(speeds) >= 2 else None,
        "avg_vss_speed": round_float(vss_speeds.mean()),
        "max_heading_change": round_float(max(heading_changes)) if heading_changes else None,
        "total_heading_change": round_float(sum(heading_changes)) if heading_changes else None,
        "brake_count": signal_count(window_df, "制动信号"),
        "left_turn_count": signal_count(window_df, "左转向灯信号"),
        "right_turn_count": signal_count(window_df, "右转向灯信号"),
        "horn_count": signal_count(window_df, "喇叭信号"),
        "reverse_count": signal_count(window_df, "倒挡信号"),
        "road_id_change_count": change_count(road_values),
        "vehicle_states": vehicle_states,
        "districts": districts,
        "data_quality_flags": data_quality_flags(window_df, speeds, headings),
    }


def stable_window_id(vehicle_id: str, start_time: str, start_index: int, point_count: int) -> str:
    raw = f"{vehicle_id}|{start_time}|{start_index}|{point_count}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)


def text_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return [clean_text(value) for value in df[column].tolist() if clean_text(value)]


def signal_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return int((values > 0).sum())


def change_count(values: list[str]) -> int:
    if len(values) < 2:
        return 0
    return sum(1 for left, right in zip(values, values[1:]) if left != right)


def circular_diffs(values: pd.Series) -> list[float]:
    if len(values) < 2:
        return []
    result = []
    for left, right in zip(values.tolist(), values.tolist()[1:]):
        raw = abs(float(right) - float(left)) % 360
        result.append(min(raw, 360 - raw))
    return result


def total_distance_km(df: pd.DataFrame) -> float | None:
    if "Lat" not in df.columns or "Lng" not in df.columns:
        return None

    coords = [
        (float(row["Lat"]), float(row["Lng"]))
        for _, row in df[["Lat", "Lng"]].dropna().iterrows()
        if -90 <= float(row["Lat"]) <= 90 and -180 <= float(row["Lng"]) <= 180
    ]
    if len(coords) < 2:
        return None

    return sum(haversine_km(coords[index], coords[index + 1]) for index in range(len(coords) - 1))


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = left
    lat2, lon2 = right
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def data_quality_flags(df: pd.DataFrame, speeds: pd.Series, headings: pd.Series) -> list[str]:
    flags = []
    if len(speeds) < len(df):
        flags.append("missing_speed")
    if len(headings) < len(df):
        flags.append("missing_heading")
    if "Lat" not in df.columns or "Lng" not in df.columns or df[["Lat", "Lng"]].dropna().empty:
        flags.append("missing_coordinates")
    if len(df) > 1:
        gaps = df["_label_time"].diff().dt.total_seconds().dropna()
        if not gaps.empty and gaps.max() > 120:
            flags.append("large_time_gap")
    return flags


def clean_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def clean_int(value: Any) -> int | None:
    number = clean_number(value)
    if number is None:
        return None
    return int(number)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def round_float(value: Any) -> float | None:
    number = clean_number(value)
    return round(number, 4) if number is not None else None


def format_time(value: Any) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
