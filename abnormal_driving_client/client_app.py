from __future__ import annotations

import csv
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import tkinter as tk
import torch
from tkinter import filedialog, messagebox, ttk

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

from model_definition import CNNLSTMModel, DEVICE, FEATURE_COLS, WINDOW_SIZE


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
SCALER_PATH = BASE_DIR / "scaler.gz"
EVENT_LOG_PATH = BASE_DIR / "events.csv"
NUM_CLASSES = 4
CONFIDENCE_THRESHOLD = 0.60
API_CALL_INTERVAL_SECONDS = 5
AMAP_REVERSE_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"

CLASS_LABELS = {
    "zh": {
        0: "正常驾驶",
        1: "高速行驶",
        2: "急加速/急减速",
        3: "曲折行驶",
    },
    "en": {
        0: "Normal driving",
        1: "High-speed driving",
        2: "Hard acceleration/deceleration",
        3: "Winding driving",
    },
}

EVENT_LOG_FIELDS = [
    "timestamp",
    "vehicle_id",
    "longitude",
    "latitude",
    "gps_speed",
    "road_speed_limit",
    "predicted_class",
    "confidence",
    "warning_source",
]

COLOR_BG = "#F5F7FB"
COLOR_SURFACE = "#FFFFFF"
COLOR_SURFACE_ALT = "#EEF3F8"
COLOR_PANEL = "#FFFFFF"
COLOR_TEXT = "#111827"
COLOR_TEXT_ON_DARK = "#111827"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#D8E0EA"
COLOR_INFO = "#2563EB"
COLOR_SUCCESS = "#16A34A"
COLOR_WARNING = "#D97706"
COLOR_ERROR = "#DC2626"
COLOR_BUTTON_TEXT = "#FFFFFF"
COLOR_BLUE_SOFT = "#EAF2FF"
COLOR_GREEN_SOFT = "#DCFCE7"
COLOR_RED_SOFT = "#FEE2E2"
COLOR_ORANGE_SOFT = "#FFEDD5"
COLOR_TEAL = "#0F766E"
COLOR_SHADOW = "#EDF2F7"
FONT_UI = ("Microsoft YaHei UI", 10)
FONT_TITLE = ("Microsoft YaHei UI", 28, "bold")
FONT_SECTION = ("Microsoft YaHei UI", 11, "bold")
FONT_STATUS = ("Microsoft YaHei UI", 10)
FONT_PREDICTION = ("Microsoft YaHei UI", 30, "bold")
FONT_DETAIL = ("Consolas", 10)

TRANSLATIONS = {
    "zh": {
        "app_title": "异常驾驶行为实时检测系统",
        "app_mark": "异常驾驶行为实时检测系统",
        "subtitle": "实时接收 MQTT 或 CSV 数据，显示模型预测、限速状态和最新数据点。",
        "language_button": "English",
        "system_status": "系统状态",
        "status_running": "运行中",
        "status_loading": "加载中",
        "status_error": "异常",
        "status_disconnected": "未连接",
        "mqtt_connection": "MQTT 连接配置",
        "broker_address": "Broker 地址",
        "port": "端口",
        "topic": "Topic 主题",
        "connection_status": "连接状态",
        "connected": "已连接",
        "not_connected": "未连接",
        "runtime_status": "运行状态",
        "status_prefix": "状态",
        "loading_model": "正在加载模型...",
        "prediction_card": "预测状态",
        "prediction_initial": "预测结果: -",
        "model_status": "模型状态",
        "model_file": "模型文件",
        "api_status": "API 状态",
        "api_address": "API 地址",
        "not_loaded": "未加载",
        "not_connected_short": "未连接",
        "none_value": "-",
        "road_speed_limit": "道路限速",
        "road_speed_limit_value": "- km/h",
        "speed_not_fetched": "道路限速: 未获取",
        "data_window": "数据滑动窗口",
        "data_window_count": "{current} / {total} 条",
        "progress_percent": "{percent}%",
        "latest_data": "最新数据（原始）",
        "clear": "清空",
        "connect_mqtt": "连接 MQTT",
        "disconnect_mqtt": "断开 MQTT",
        "select_csv": "选择 CSV",
        "start_simulation": "开始模拟",
        "stop_simulation": "停止模拟",
        "detail_waiting": "等待数据。可以连接 MQTT，或选择 CSV 文件进行模拟。\n",
        "model_window": "模型窗口: {current}/{total}",
        "model_load_failed": "模型加载失败: {error}",
        "load_error_title": "加载错误",
        "api_enabled": "已启用地图 API",
        "api_disabled": "未配置 AMAP_API_KEY，已关闭地图限速",
        "model_loaded": "模型和 scaler 加载成功。{api_status}",
        "file_dialog_title": "选择 GPS 特征 CSV 文件",
        "csv_file_type": "CSV 文件",
        "all_file_type": "所有文件",
        "csv_selected": "已选择 CSV: {filename}",
        "info_title": "提示",
        "simulation_already_running": "模拟已经在运行。",
        "select_csv_first": "请先选择 CSV 文件。",
        "csv_simulating": "CSV 模拟进行中",
        "csv_failed": "CSV 模拟失败: {error}",
        "csv_stopped": "CSV 模拟已停止",
        "csv_completed": "CSV 模拟完成",
        "mqtt_unavailable_title": "MQTT 不可用",
        "mqtt_unavailable": "请先安装 paho-mqtt。",
        "mqtt_already_connected": "MQTT 已连接。",
        "input_error_title": "输入错误",
        "port_must_be_number": "端口必须是数字。",
        "broker_topic_required": "Broker 地址和 Topic 不能为空。",
        "mqtt_connecting": "正在连接 MQTT: {broker}:{port}",
        "mqtt_connect_failed": "MQTT 连接失败: {error}",
        "mqtt_connect_error_title": "MQTT 连接错误",
        "mqtt_connected": "MQTT 已连接，订阅 Topic: {topic}",
        "mqtt_connect_failed_code": "MQTT 连接失败，返回码: {code}",
        "mqtt_parse_failed": "MQTT 数据解析失败: {error}",
        "mqtt_disconnected": "MQTT 已断开",
        "model_not_loaded": "模型未加载，无法预测",
        "invalid_input_data": "输入数据格式无效",
        "prediction_failed": "预测失败: {error}",
        "prediction_api_wait": "预测结果: API 超速，等待模型窗口填满 ({current}/{total})",
        "prediction_wait": "预测结果: 等待窗口填满 ({current}/{total})",
        "prediction_api_plus": "预测结果: API 超速 + {label} (置信度 {confidence:.2f})",
        "prediction_uncertain": "预测结果: 不确定 ({label}, 置信度 {confidence:.2f})",
        "prediction_result": "预测结果: {label} (置信度 {confidence:.2f})",
        "unknown": "未知",
        "speed_no_api_key": "道路限速: 未配置 AMAP_API_KEY",
        "speed_api_failed": "道路限速: API 获取失败 ({error})",
        "speed_pending": "道路限速: 正在获取或暂无结果",
        "speed_current": "道路限速: {limit} km/h，当前速度: {speed}",
        "latest_data_point": "最新数据点",
        "buffer": "缓冲区",
        "vehicle_id": "车辆 ID",
        "gps_time": "GPS 时间",
        "gps_speed": "gps速度",
        "vss_speed": "vss速度",
        "acc_status": "ACC状态",
        "model_result": "模型结果: {label}，置信度 {confidence:.2f}",
        "api_overspeed_detail": "API 判断: 当前速度超过道路限速 10%",
        "warning_source": "预警来源",
        "data_notes": "数据提示:",
        "missing_fields": "缺少模型字段: {fields}",
        "invalid_field": "字段 {field} 不是有效数字，已使用 0",
    },
    "en": {
        "app_title": "Real-Time Abnormal Driving Detection",
        "app_mark": "Real-Time Abnormal Driving Detection",
        "subtitle": "Receive MQTT or CSV data, then show model prediction, speed-limit status, and the latest data point.",
        "language_button": "中文",
        "system_status": "System Status",
        "status_running": "Running",
        "status_loading": "Loading",
        "status_error": "Error",
        "status_disconnected": "Disconnected",
        "mqtt_connection": "MQTT Connection Config",
        "broker_address": "Broker address",
        "port": "Port",
        "topic": "Topic",
        "connection_status": "Connection status",
        "connected": "Connected",
        "not_connected": "Not connected",
        "runtime_status": "Runtime Status",
        "status_prefix": "Status",
        "loading_model": "Loading model...",
        "prediction_card": "Prediction Status",
        "prediction_initial": "Prediction: -",
        "model_status": "Model status",
        "model_file": "Model file",
        "api_status": "API status",
        "api_address": "API address",
        "not_loaded": "Not loaded",
        "not_connected_short": "Not connected",
        "none_value": "-",
        "road_speed_limit": "Road Speed Limit",
        "road_speed_limit_value": "- km/h",
        "speed_not_fetched": "Road speed limit: not fetched",
        "data_window": "Data sliding window",
        "data_window_count": "{current} / {total} rows",
        "progress_percent": "{percent}%",
        "latest_data": "Latest Data (Raw)",
        "clear": "Clear",
        "connect_mqtt": "Connect MQTT",
        "disconnect_mqtt": "Disconnect MQTT",
        "select_csv": "Select CSV",
        "start_simulation": "Start Simulation",
        "stop_simulation": "Stop Simulation",
        "detail_waiting": "Waiting for data. Connect MQTT or select a CSV file to simulate data.\n",
        "model_window": "Model window: {current}/{total}",
        "model_load_failed": "Model load failed: {error}",
        "load_error_title": "Load Error",
        "api_enabled": "Map API enabled",
        "api_disabled": "AMAP_API_KEY is not set. Map speed-limit checks are off",
        "model_loaded": "Model and scaler loaded. {api_status}",
        "file_dialog_title": "Select GPS feature CSV file",
        "csv_file_type": "CSV files",
        "all_file_type": "All files",
        "csv_selected": "Selected CSV: {filename}",
        "info_title": "Notice",
        "simulation_already_running": "Simulation is already running.",
        "select_csv_first": "Please select a CSV file first.",
        "csv_simulating": "CSV simulation is running",
        "csv_failed": "CSV simulation failed: {error}",
        "csv_stopped": "CSV simulation stopped",
        "csv_completed": "CSV simulation completed",
        "mqtt_unavailable_title": "MQTT Unavailable",
        "mqtt_unavailable": "Please install paho-mqtt first.",
        "mqtt_already_connected": "MQTT is already connected.",
        "input_error_title": "Input Error",
        "port_must_be_number": "Port must be a number.",
        "broker_topic_required": "Broker address and Topic cannot be empty.",
        "mqtt_connecting": "Connecting to MQTT: {broker}:{port}",
        "mqtt_connect_failed": "MQTT connection failed: {error}",
        "mqtt_connect_error_title": "MQTT Connection Error",
        "mqtt_connected": "MQTT connected. Subscribed to Topic: {topic}",
        "mqtt_connect_failed_code": "MQTT connection failed. Return code: {code}",
        "mqtt_parse_failed": "Failed to parse MQTT data: {error}",
        "mqtt_disconnected": "MQTT disconnected",
        "model_not_loaded": "Model is not loaded. Prediction is unavailable",
        "invalid_input_data": "Input data format is invalid",
        "prediction_failed": "Prediction failed: {error}",
        "prediction_api_wait": "Prediction: API overspeed. Waiting for model window ({current}/{total})",
        "prediction_wait": "Prediction: waiting for model window ({current}/{total})",
        "prediction_api_plus": "Prediction: API overspeed + {label} (confidence {confidence:.2f})",
        "prediction_uncertain": "Prediction: uncertain ({label}, confidence {confidence:.2f})",
        "prediction_result": "Prediction: {label} (confidence {confidence:.2f})",
        "unknown": "Unknown",
        "speed_no_api_key": "Road speed limit: AMAP_API_KEY is not set",
        "speed_api_failed": "Road speed limit: API request failed ({error})",
        "speed_pending": "Road speed limit: fetching or no result yet",
        "speed_current": "Road speed limit: {limit} km/h, current speed: {speed}",
        "latest_data_point": "Latest Data Point",
        "buffer": "Buffer",
        "vehicle_id": "Vehicle ID",
        "gps_time": "GPS time",
        "gps_speed": "GPS speed",
        "vss_speed": "VSS speed",
        "acc_status": "ACC status",
        "model_result": "Model result: {label}, confidence {confidence:.2f}",
        "api_overspeed_detail": "API check: current speed is more than 10% over the road speed limit",
        "warning_source": "Warning source",
        "data_notes": "Data notes:",
        "missing_fields": "Missing model fields: {fields}",
        "invalid_field": "Field {field} is not a valid number. Used 0 instead",
    },
}


def load_model_and_scaler() -> tuple[CNNLSTMModel, Any]:
    """Load the trained model and scaler from the client folder."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    model = CNNLSTMModel(input_size=len(FEATURE_COLS), num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    expected_features = getattr(scaler, "n_features_in_", len(FEATURE_COLS))
    if expected_features != len(FEATURE_COLS):
        raise ValueError(
            f"Scaler expects {expected_features} features, but model input has {len(FEATURE_COLS)}."
        )

    return model, scaler


def parse_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_acc_status(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return 1.0 if normalized in {"acc开", "acc_on", "on", "1", "true"} else 0.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        return 1.0 if float(value) == 1.0 else 0.0
    return 0.0


def build_feature_vector(raw_data: dict[str, Any], language: str = "zh") -> tuple[list[float] | None, list[str]]:
    """Convert one raw data point into the exact model feature order."""
    messages = TRANSLATIONS.get(language, TRANSLATIONS["zh"])
    missing = [col for col in FEATURE_COLS if col not in raw_data]
    if missing:
        return None, [messages["missing_fields"].format(fields=", ".join(missing))]

    values: list[float] = []
    warnings: list[str] = []
    for col in FEATURE_COLS:
        if col == "ACC状态":
            values.append(parse_acc_status(raw_data.get(col)))
            continue

        parsed = parse_float(raw_data.get(col))
        if parsed is None:
            warnings.append(messages["invalid_field"].format(field=col))
            parsed = 0.0
        values.append(parsed)

    return values, warnings


def predict_behavior(
    model: CNNLSTMModel,
    scaler: Any,
    window_values: list[list[float]],
) -> tuple[int, float]:
    """Run one model prediction from the current sliding window."""
    window_frame = pd.DataFrame(window_values, columns=FEATURE_COLS, dtype=np.float32)
    scaled_window = scaler.transform(window_frame)
    input_tensor = torch.FloatTensor(scaled_window).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())

    return predicted_index, confidence


def get_speed_limit_from_amap(api_key: str, latitude: float, longitude: float) -> int | None:
    params = {
        "key": api_key,
        "location": f"{longitude},{latitude}",
        "extensions": "base",
        "radius": "1000",
        "roadlevel": "1",
    }
    response = requests.get(AMAP_REVERSE_GEOCODE_URL, params=params, timeout=3)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "1":
        return None

    regeocode = data.get("regeocode", {})
    for road_group in ("roads", "roadinters"):
        for item in regeocode.get(road_group, []) or []:
            speed = item.get("speed")
            if isinstance(speed, str) and speed.isdigit():
                return int(speed)
            if isinstance(speed, (int, float)):
                return int(speed)
    return None


def append_event_log(row: dict[str, Any]) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = EVENT_LOG_PATH.exists()
    with EVENT_LOG_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in EVENT_LOG_FIELDS})


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.language = "zh"
        self.root.title(self._t("app_title"))
        self.root.geometry("1180x820")
        self.root.minsize(1040, 720)
        self.root.configure(bg=COLOR_BG)

        self.model: CNNLSTMModel | None = None
        self.scaler: Any | None = None
        self.data_buffer: deque[list[float]] = deque(maxlen=WINDOW_SIZE)
        self.mqtt_client: Any | None = None
        self.is_mqtt_connected = False
        self.is_simulating = False
        self.stop_simulation_event = threading.Event()
        self.simulation_thread: threading.Thread | None = None
        self.csv_file_path: str | None = None
        self.simulation_delay = 0.2

        self.amap_api_key = os.getenv("AMAP_API_KEY", "").strip()
        self.api_queue: Queue[tuple[str, int | None, str | None]] = Queue()
        self.api_request_running = False
        self.last_api_call_time = 0.0
        self.current_road_speed_limit: int | None = None
        self.last_api_error: str | None = None
        self.last_status_key = "loading_model"
        self.last_status_params: dict[str, Any] = {}
        self.last_status_color = COLOR_MUTED
        self.last_prediction_state: tuple[tuple[int, float] | None, bool] | None = None
        self.last_speed_state: tuple[float | None, bool] | None = None
        self.last_detail_state: tuple[
            dict[str, Any],
            list[str],
            tuple[int, float] | None,
            bool | None,
            str | None,
        ] | None = None

        self._build_ui()
        self._load_runtime_assets()
        self.root.after(300, self._poll_api_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_ui(self) -> None:
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure(
            "TEntry",
            fieldbackground=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            padding=8,
        )
        self.style.map("TEntry", bordercolor=[("focus", COLOR_INFO)])
        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor="#E5EAF1",
            background=COLOR_SUCCESS,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_SUCCESS,
            darkcolor=COLOR_SUCCESS,
        )

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        shell = tk.Frame(self.root, bg=COLOR_BG)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=1)

        app_bar = tk.Frame(shell, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER, highlightthickness=1)
        app_bar.grid(row=0, column=0, sticky="ew")
        app_bar.columnconfigure(1, weight=1)
        self._icon_label(app_bar, "●", self._t("app_mark"), COLOR_INFO, font=("Microsoft YaHei UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=18, pady=12
        )
        self.language_button = ttk.Button(
            app_bar,
            text=self._t("language_button"),
            command=self.toggle_language,
        )
        self.language_button.grid(row=0, column=2, sticky="e", padx=(0, 18), pady=10)

        title_bar = tk.Frame(shell, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER, highlightthickness=1)
        title_bar.grid(row=1, column=0, sticky="ew")
        title_bar.columnconfigure(0, weight=1)
        title_bar.columnconfigure(1, weight=0)
        title_bar.columnconfigure(2, weight=1)
        self.title_label = tk.Label(
            title_bar,
            text=self._t("app_title"),
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            font=FONT_TITLE,
        )
        self.title_label.grid(row=0, column=1, sticky="n", pady=(24, 6))
        self.subtitle_label = tk.Label(
            title_bar,
            text=self._t("subtitle"),
            bg=COLOR_SURFACE,
            fg=COLOR_MUTED,
            font=FONT_STATUS,
        )
        self.subtitle_label.grid(row=1, column=1, sticky="n", pady=(0, 24))
        self.system_status_pill = self._make_pill(
            title_bar,
            self._t("system_status"),
            self._t("status_loading"),
            COLOR_SUCCESS,
            COLOR_GREEN_SOFT,
        )
        self.system_status_pill.grid(row=0, column=2, rowspan=2, sticky="e", padx=26)

        content = tk.Frame(shell, bg=COLOR_BG)
        content.grid(row=2, column=0, sticky="ew", padx=22, pady=(18, 18))
        content.columnconfigure(0, minsize=330)
        content.columnconfigure(1, weight=1)

        mqtt_card = self._create_card(content)
        mqtt_card.grid(row=0, column=0, sticky="nsew", padx=(0, 22))
        mqtt_card.columnconfigure(0, weight=1)
        self.mqtt_title_label = self._section_header(mqtt_card, "↗", self._t("mqtt_connection"))
        self.mqtt_title_label.grid(row=0, column=0, sticky="w", padx=22, pady=(22, 18))

        self.broker_label = self._form_label(mqtt_card, self._t("broker_address"))
        self.broker_label.grid(row=1, column=0, sticky="w", padx=22, pady=(0, 8))
        self.broker_entry = ttk.Entry(mqtt_card)
        self.broker_entry.insert(0, "localhost")
        self.broker_entry.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 20), ipady=5)

        self.port_label = self._form_label(mqtt_card, self._t("port"))
        self.port_label.grid(row=3, column=0, sticky="w", padx=22, pady=(0, 8))
        self.port_entry = ttk.Entry(mqtt_card)
        self.port_entry.insert(0, "1883")
        self.port_entry.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 20), ipady=5)

        self.topic_label = self._form_label(mqtt_card, self._t("topic"))
        self.topic_label.grid(row=5, column=0, sticky="w", padx=22, pady=(0, 8))
        self.topic_entry = ttk.Entry(mqtt_card)
        self.topic_entry.insert(0, "vehicle/gps_data")
        self.topic_entry.grid(row=6, column=0, sticky="ew", padx=22, pady=(0, 22), ipady=5)

        separator = tk.Frame(mqtt_card, bg=COLOR_BORDER, height=1)
        separator.grid(row=7, column=0, sticky="ew", padx=22, pady=(0, 18))
        connection_row = tk.Frame(mqtt_card, bg=COLOR_SURFACE)
        connection_row.grid(row=8, column=0, sticky="ew", padx=22, pady=(0, 22))
        connection_row.columnconfigure(0, weight=1)
        self.connection_status_label = tk.Label(
            connection_row,
            text=self._t("connection_status"),
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.connection_status_label.grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.connection_status_pill = self._make_value_pill(connection_row, self._t("not_connected"), COLOR_MUTED, COLOR_SURFACE_ALT)
        self.connection_status_pill.grid(row=0, column=1, sticky="e")

        prediction_card = self._create_card(content)
        prediction_card.grid(row=0, column=1, sticky="nsew")
        prediction_card.columnconfigure(0, weight=1)
        self.prediction_title_label = self._section_header(prediction_card, "▮", self._t("prediction_card"))
        self.prediction_title_label.grid(row=0, column=0, sticky="w", padx=26, pady=(22, 14))

        prediction_body = tk.Frame(prediction_card, bg=COLOR_SURFACE)
        prediction_body.grid(row=1, column=0, sticky="nsew", padx=26, pady=(0, 20))
        prediction_body.columnconfigure(1, weight=1)
        self.steering_canvas = self._create_steering_icon(prediction_body)
        self.steering_canvas.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 34), pady=(18, 0))
        self.prediction_label = tk.Label(
            prediction_body,
            text=self._t("prediction_initial"),
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            font=FONT_PREDICTION,
            anchor="w",
        )
        self.prediction_label.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(28, 16))
        divider = tk.Frame(prediction_body, bg=COLOR_BORDER, height=1)
        divider.grid(row=1, column=1, columnspan=2, sticky="ew")

        self.model_status_label, self.model_status_value, self.model_file_label = self._status_row(
            prediction_body,
            row=2,
            icon="◇",
            label_key="model_status",
            value_text=self._t("not_loaded"),
            value_color=COLOR_INFO,
            soft_color=COLOR_BLUE_SOFT,
            extra_key="model_file",
        )
        self.api_status_label, self.api_status_value, self.api_address_label = self._status_row(
            prediction_body,
            row=3,
            icon="API",
            label_key="api_status",
            value_text=self._t("not_connected_short"),
            value_color=COLOR_INFO,
            soft_color=COLOR_BLUE_SOFT,
            extra_key="api_address",
        )
        self.road_speed_label, self.road_speed_value, _ = self._status_row(
            prediction_body,
            row=4,
            icon="◷",
            label_key="road_speed_limit",
            value_text=self._t("road_speed_limit_value"),
            value_color=COLOR_TEXT,
            soft_color=COLOR_SURFACE,
        )
        self.status_label = tk.Label(
            prediction_body,
            text=f"{self._t('status_prefix')}: {self._t('loading_model')}",
            bg=COLOR_SURFACE,
            fg=COLOR_MUTED,
            font=FONT_STATUS,
            anchor="w",
        )
        self.status_label.grid(row=5, column=1, columnspan=2, sticky="ew", pady=(16, 0))

        window_bar = tk.Frame(prediction_card, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER, highlightthickness=1)
        window_bar.grid(row=2, column=0, sticky="ew")
        window_bar.columnconfigure(1, weight=1)
        self.window_label = self._icon_text(window_bar, "○", self._t("data_window"), COLOR_INFO)
        self.window_label.grid(row=0, column=0, sticky="w", padx=26, pady=(18, 8))
        self.buffer_count_label = tk.Label(
            window_bar,
            text=self._t("data_window_count", current=0, total=WINDOW_SIZE),
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.buffer_count_label.grid(row=0, column=2, sticky="e", padx=26, pady=(18, 8))
        self.buffer_progress = ttk.Progressbar(
            window_bar,
            maximum=WINDOW_SIZE,
            value=0,
            mode="determinate",
            style="Modern.Horizontal.TProgressbar",
        )
        self.buffer_progress.grid(row=1, column=0, columnspan=3, sticky="ew", padx=26, pady=(0, 8))
        self.buffer_label = tk.Label(
            window_bar,
            text=self._t("progress_percent", percent=0),
            bg=COLOR_SURFACE,
            fg=COLOR_MUTED,
            font=FONT_STATUS,
        )
        self.buffer_label.grid(row=2, column=0, columnspan=3, sticky="n", pady=(0, 16))

        detail_frame = self._create_card(shell)
        detail_frame.grid(row=3, column=0, sticky="nsew", padx=22, pady=(0, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=1)
        detail_header = tk.Frame(detail_frame, bg=COLOR_SURFACE)
        detail_header.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 12))
        detail_header.columnconfigure(0, weight=1)
        self.detail_title_label = self._section_header(detail_header, "▤", self._t("latest_data"))
        self.detail_title_label.grid(row=0, column=0, sticky="w")
        self.clear_button = self._outline_button(
            detail_header,
            self._t("clear"),
            self.clear_detail_panel,
            COLOR_MUTED,
            COLOR_BORDER,
            width=8,
        )
        self.clear_button.grid(row=0, column=1, sticky="e")

        self.detail_text = tk.Text(
            detail_frame,
            height=18,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_ON_DARK,
            insertbackground=COLOR_TEXT_ON_DARK,
            font=FONT_DETAIL,
            relief="solid",
            borderwidth=0,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=14,
            pady=12,
            wrap="word",
        )
        self.detail_text.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 22))
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_text.yview)
        detail_scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 22))
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)
        self._write_detail(self._t("detail_waiting"))

        button_frame = tk.Frame(shell, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER, highlightthickness=1)
        button_frame.grid(row=4, column=0, sticky="ew")
        for column in range(5):
            button_frame.columnconfigure(column, weight=1)

        self.connect_button = self._outline_button(button_frame, "↗  " + self._t("connect_mqtt"), self.connect_mqtt, COLOR_INFO)
        self.connect_button.grid(row=0, column=0, padx=(22, 10), pady=18, sticky="ew", ipady=8)
        self.disconnect_button = self._outline_button(
            button_frame,
            "↘  " + self._t("disconnect_mqtt"),
            self.disconnect_mqtt,
            COLOR_ERROR,
            state=tk.DISABLED,
        )
        self.disconnect_button.grid(row=0, column=1, padx=10, pady=18, sticky="ew", ipady=8)

        self.select_file_button = self._outline_button(button_frame, "□  " + self._t("select_csv"), self.select_csv_file, COLOR_TEAL)
        self.select_file_button.grid(row=0, column=2, padx=10, pady=18, sticky="ew", ipady=8)
        self.start_simulation_button = self._outline_button(
            button_frame,
            "▷  " + self._t("start_simulation"),
            self.start_simulation,
            COLOR_INFO,
            state=tk.DISABLED,
        )
        self.start_simulation_button.grid(row=0, column=3, padx=10, pady=18, sticky="ew", ipady=8)
        self.stop_simulation_button = self._outline_button(
            button_frame,
            "■  " + self._t("stop_simulation"),
            self.stop_simulation,
            COLOR_WARNING,
            state=tk.DISABLED,
        )
        self.stop_simulation_button.grid(row=0, column=4, padx=(10, 22), pady=18, sticky="ew", ipady=8)

    def _create_card(self, parent: tk.Misc) -> tk.Frame:
        return tk.Frame(parent, bg=COLOR_SURFACE, highlightbackground=COLOR_BORDER, highlightthickness=1)

    def _icon_label(self, parent: tk.Misc, icon: str, text: str, color: str, font: tuple[str, int, str]) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLOR_SURFACE)
        tk.Label(frame, text=icon, bg=COLOR_SURFACE, fg=color, font=font).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(frame, text=text, bg=COLOR_SURFACE, fg=COLOR_TEXT, font=font).pack(side=tk.LEFT)
        return frame

    def _section_header(self, parent: tk.Misc, icon: str, text: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLOR_SURFACE)
        tk.Label(frame, text=icon, bg=COLOR_SURFACE, fg=COLOR_INFO, font=("Microsoft YaHei UI", 15, "bold")).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        label = tk.Label(frame, text=text, bg=COLOR_SURFACE, fg=COLOR_TEXT, font=("Microsoft YaHei UI", 14, "bold"))
        label.pack(side=tk.LEFT)
        frame.text_label = label  # type: ignore[attr-defined]
        return frame

    def _set_section_text(self, frame: tk.Frame, text: str) -> None:
        getattr(frame, "text_label").config(text=text)

    def _form_label(self, parent: tk.Misc, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=COLOR_SURFACE, fg=COLOR_TEXT, font=("Microsoft YaHei UI", 11), anchor="w")

    def _icon_text(self, parent: tk.Misc, icon: str, text: str, color: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLOR_SURFACE)
        tk.Label(frame, text=icon, bg=COLOR_SURFACE, fg=color, font=("Microsoft YaHei UI", 12, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        label = tk.Label(frame, text=text, bg=COLOR_SURFACE, fg=COLOR_TEXT, font=("Microsoft YaHei UI", 11, "bold"))
        label.pack(side=tk.LEFT)
        frame.text_label = label  # type: ignore[attr-defined]
        return frame

    def _status_row(
        self,
        parent: tk.Misc,
        row: int,
        icon: str,
        label_key: str,
        value_text: str,
        value_color: str,
        soft_color: str,
        extra_key: str | None = None,
    ) -> tuple[tk.Label, tk.Label, tk.Label | None]:
        row_frame = tk.Frame(parent, bg=COLOR_SURFACE)
        row_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=(16, 0))
        row_frame.columnconfigure(2, weight=1)

        tk.Label(row_frame, text=icon, bg=COLOR_SURFACE, fg=COLOR_INFO, font=("Microsoft YaHei UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        label = tk.Label(
            row_frame,
            text=self._t(label_key) + ":",
            bg=COLOR_SURFACE,
            fg=COLOR_TEXT,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        label.grid(row=0, column=1, sticky="w", padx=(0, 10))
        value = tk.Label(
            row_frame,
            text=value_text,
            bg=soft_color,
            fg=value_color,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=10,
            pady=4,
        )
        value.grid(row=0, column=2, sticky="w")

        extra_label = None
        if extra_key:
            extra_label = tk.Label(
                row_frame,
                text=self._t(extra_key) + ": " + self._t("none_value"),
                bg=COLOR_SURFACE,
                fg=COLOR_TEXT,
                font=("Microsoft YaHei UI", 11, "bold"),
            )
            extra_label.grid(row=0, column=3, sticky="e", padx=(24, 0))

        return label, value, extra_label

    def _outline_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        color: str,
        border_color: str | None = None,
        width: int = 16,
        state: str = tk.NORMAL,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            state=state,
            bg=COLOR_SURFACE,
            fg=color,
            activebackground=COLOR_BLUE_SOFT if color == COLOR_INFO else COLOR_SURFACE_ALT,
            activeforeground=color,
            disabledforeground="#94A3B8",
            font=("Microsoft YaHei UI", 12, "bold"),
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=border_color or color,
            cursor="hand2",
        )

    def _make_value_pill(self, parent: tk.Misc, text: str, color: str, bg: str) -> tk.Label:
        return tk.Label(
            parent,
            text="● " + text,
            bg=bg,
            fg=color,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=10,
            pady=4,
        )

    def _make_pill(self, parent: tk.Misc, label: str, value: str, color: str, bg: str) -> tk.Label:
        return tk.Label(
            parent,
            text=f"● {label}: {value}",
            bg=bg,
            fg=color,
            font=("Microsoft YaHei UI", 11, "bold"),
            padx=16,
            pady=9,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )

    def _create_steering_icon(self, parent: tk.Misc) -> tk.Canvas:
        canvas = tk.Canvas(parent, width=118, height=118, bg=COLOR_SURFACE, highlightthickness=0)
        canvas.create_oval(8, 8, 110, 110, fill=COLOR_BLUE_SOFT, outline="")
        canvas.create_oval(30, 30, 88, 88, outline=COLOR_INFO, width=7)
        canvas.create_oval(54, 54, 64, 64, fill=COLOR_INFO, outline=COLOR_INFO)
        canvas.create_line(59, 59, 59, 34, fill=COLOR_INFO, width=6, capstyle=tk.ROUND)
        canvas.create_line(59, 59, 37, 76, fill=COLOR_INFO, width=6, capstyle=tk.ROUND)
        canvas.create_line(59, 59, 81, 76, fill=COLOR_INFO, width=6, capstyle=tk.ROUND)
        return canvas

    def _t(self, key: str, **kwargs: Any) -> str:
        text = TRANSLATIONS.get(self.language, TRANSLATIONS["zh"]).get(key, key)
        return text.format(**kwargs) if kwargs else text

    def _class_label(self, class_index: int) -> str:
        return CLASS_LABELS.get(self.language, CLASS_LABELS["zh"]).get(class_index, self._t("unknown"))

    def toggle_language(self) -> None:
        self.language = "en" if self.language == "zh" else "zh"
        self._apply_language()

    def _apply_language(self) -> None:
        self.root.title(self._t("app_title"))
        self.title_label.config(text=self._t("app_title"))
        self.subtitle_label.config(text=self._t("subtitle"))
        self.language_button.config(text=self._t("language_button"))
        self._set_section_text(self.mqtt_title_label, self._t("mqtt_connection"))
        self.broker_label.config(text=self._t("broker_address"))
        self.port_label.config(text=self._t("port"))
        self.topic_label.config(text=self._t("topic"))
        self.connection_status_label.config(text=self._t("connection_status"))
        self._set_connection_state(self.is_mqtt_connected)
        self._set_section_text(self.prediction_title_label, self._t("prediction_card"))
        self.model_status_label.config(text=self._t("model_status") + ":")
        self.api_status_label.config(text=self._t("api_status") + ":")
        self.road_speed_label.config(text=self._t("road_speed_limit") + ":")
        if self.model_file_label is not None:
            self.model_file_label.config(text=self._t("model_file") + ": " + MODEL_PATH.name)
        if self.api_address_label is not None:
            self.api_address_label.config(text=self._t("api_address") + ": " + AMAP_REVERSE_GEOCODE_URL)
        self._set_section_text(self.window_label, self._t("data_window"))
        self._set_section_text(self.detail_title_label, self._t("latest_data"))
        self.clear_button.config(text=self._t("clear"))
        self.connect_button.config(text="↗  " + self._t("connect_mqtt"))
        self.disconnect_button.config(text="↘  " + self._t("disconnect_mqtt"))
        self.select_file_button.config(text="□  " + self._t("select_csv"))
        self.start_simulation_button.config(text="▷  " + self._t("start_simulation"))
        self.stop_simulation_button.config(text="■  " + self._t("stop_simulation"))
        self._refresh_status()
        self._update_buffer_progress()
        self._refresh_model_api_rows()

        if self.last_prediction_state is None:
            self.prediction_label.config(text=self._t("prediction_initial"), foreground=COLOR_MUTED)
        else:
            prediction, api_overspeed = self.last_prediction_state
            display_label, _, color = self._build_prediction_display(prediction, api_overspeed)
            self.prediction_label.config(text=display_label, foreground=color)

        if self.last_speed_state is None:
            self.road_speed_value.config(text=self._t("road_speed_limit_value"), fg=COLOR_TEXT, bg=COLOR_SURFACE)
        else:
            gps_speed, api_overspeed = self.last_speed_state
            self._update_speed_limit_label(gps_speed, api_overspeed, remember=False)

        if self.last_detail_state is None:
            self._write_detail(self._t("detail_waiting"))
        else:
            raw_data, warnings, prediction, api_overspeed, warning_source = self.last_detail_state
            self._update_detail_panel(raw_data, warnings, prediction, api_overspeed, warning_source, remember=False)

    def _load_runtime_assets(self) -> None:
        try:
            self.model, self.scaler = load_model_and_scaler()
        except Exception as exc:
            self._set_status("model_load_failed", COLOR_ERROR, error=exc)
            self._refresh_model_api_rows(model_loaded=False)
            messagebox.showerror(self._t("load_error_title"), str(exc))
            self._set_controls_enabled(False)
            return

        api_status_key = "api_enabled" if self.amap_api_key else "api_disabled"
        self._set_status("model_loaded", COLOR_SUCCESS, api_status_key=api_status_key)
        self._refresh_model_api_rows(model_loaded=True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.connect_button.config(state=state if mqtt else tk.DISABLED)
        self.select_file_button.config(state=state)

    def _set_status(self, message_key: str, color: str = COLOR_MUTED, **params: Any) -> None:
        self.last_status_key = message_key
        self.last_status_params = params
        self.last_status_color = color
        self._refresh_status()

    def _refresh_status(self) -> None:
        params = dict(self.last_status_params)
        if "api_status_key" in params:
            params["api_status"] = self._t(params.pop("api_status_key"))
        message = self._t(self.last_status_key, **params)
        self.status_label.config(text=f"{self._t('status_prefix')}: {message}", foreground=self.last_status_color)
        self._update_system_status_pill(self.last_status_color)

    def _update_buffer_progress(self) -> None:
        current_size = len(self.data_buffer)
        percent = int((current_size / WINDOW_SIZE) * 100) if WINDOW_SIZE else 0
        self.buffer_count_label.config(text=self._t("data_window_count", current=current_size, total=WINDOW_SIZE))
        self.buffer_label.config(text=self._t("progress_percent", percent=percent))
        self.buffer_progress.config(value=current_size)

    def _update_system_status_pill(self, color: str) -> None:
        if color == COLOR_ERROR:
            value = self._t("status_error")
            fg = COLOR_ERROR
            bg = COLOR_RED_SOFT
        elif color == COLOR_WARNING:
            value = self._t("status_disconnected")
            fg = COLOR_WARNING
            bg = COLOR_ORANGE_SOFT
        elif color == COLOR_SUCCESS:
            value = self._t("status_running")
            fg = COLOR_SUCCESS
            bg = COLOR_GREEN_SOFT
        else:
            value = self._t("status_loading")
            fg = COLOR_INFO
            bg = COLOR_BLUE_SOFT
        self.system_status_pill.config(text=f"● {self._t('system_status')}: {value}", fg=fg, bg=bg)

    def _set_connection_state(self, connected: bool) -> None:
        if connected:
            self.connection_status_pill.config(text="● " + self._t("connected"), fg=COLOR_SUCCESS, bg=COLOR_GREEN_SOFT)
        else:
            self.connection_status_pill.config(text="● " + self._t("not_connected"), fg=COLOR_MUTED, bg=COLOR_SURFACE_ALT)

    def _refresh_model_api_rows(self, model_loaded: bool | None = None) -> None:
        if model_loaded is None:
            model_loaded = self.model is not None and self.scaler is not None
        self.model_status_value.config(
            text=self._t("status_running") if model_loaded else self._t("not_loaded"),
            fg=COLOR_SUCCESS if model_loaded else COLOR_INFO,
            bg=COLOR_GREEN_SOFT if model_loaded else COLOR_BLUE_SOFT,
        )
        self.api_status_value.config(
            text=self._t("api_enabled") if self.amap_api_key else self._t("not_connected_short"),
            fg=COLOR_SUCCESS if self.amap_api_key else COLOR_INFO,
            bg=COLOR_GREEN_SOFT if self.amap_api_key else COLOR_BLUE_SOFT,
        )
        if self.model_file_label is not None:
            self.model_file_label.config(text=self._t("model_file") + ": " + MODEL_PATH.name)
        if self.api_address_label is not None:
            api_text = AMAP_REVERSE_GEOCODE_URL if self.amap_api_key else self._t("none_value")
            self.api_address_label.config(text=self._t("api_address") + ": " + api_text)

    def _write_detail(self, text: str) -> None:
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text)
        self.detail_text.config(state=tk.DISABLED)

    def clear_detail_panel(self) -> None:
        self.last_detail_state = None
        self._write_detail(self._t("detail_waiting"))

    def select_csv_file(self) -> None:
        path = filedialog.askopenfilename(
            title=self._t("file_dialog_title"),
            filetypes=((self._t("csv_file_type"), "*.csv"), (self._t("all_file_type"), "*.*")),
        )
        if not path:
            return
        self.csv_file_path = path
        self.start_simulation_button.config(state=tk.NORMAL)
        self._set_status("csv_selected", COLOR_INFO, filename=Path(path).name)

    def start_simulation(self) -> None:
        if self.is_simulating:
            messagebox.showinfo(self._t("info_title"), self._t("simulation_already_running"))
            return
        if not self.csv_file_path:
            messagebox.showwarning(self._t("info_title"), self._t("select_csv_first"))
            return

        self.data_buffer.clear()
        self._update_buffer_progress()
        self.stop_simulation_event.clear()
        self.is_simulating = True
        self.start_simulation_button.config(state=tk.DISABLED)
        self.stop_simulation_button.config(state=tk.NORMAL)
        self.select_file_button.config(state=tk.DISABLED)
        self.simulation_thread = threading.Thread(target=self._simulation_worker, daemon=True)
        self.simulation_thread.start()
        self._set_status("csv_simulating", COLOR_INFO)

    def _simulation_worker(self) -> None:
        try:
            assert self.csv_file_path is not None
            data = pd.read_csv(self.csv_file_path)
            for _, row in data.iterrows():
                if self.stop_simulation_event.is_set():
                    break
                self.root.after(0, self.process_data_point, row.to_dict())
                time.sleep(self.simulation_delay)
        except Exception as exc:
            self.root.after(0, self._set_status, "csv_failed", COLOR_ERROR, error=exc)
        finally:
            self.root.after(0, self._finish_simulation)

    def _finish_simulation(self) -> None:
        self.is_simulating = False
        self.start_simulation_button.config(state=tk.NORMAL if self.csv_file_path else tk.DISABLED)
        self.stop_simulation_button.config(state=tk.DISABLED)
        self.select_file_button.config(state=tk.NORMAL)
        if self.stop_simulation_event.is_set():
            self._set_status("csv_stopped", COLOR_WARNING)
        else:
            self._set_status("csv_completed", COLOR_SUCCESS)

    def stop_simulation(self) -> None:
        self.stop_simulation_event.set()

    def connect_mqtt(self) -> None:
        if mqtt is None:
            messagebox.showerror(self._t("mqtt_unavailable_title"), self._t("mqtt_unavailable"))
            return
        if self.is_mqtt_connected:
            messagebox.showinfo(self._t("info_title"), self._t("mqtt_already_connected"))
            return

        broker = self.broker_entry.get().strip()
        topic = self.topic_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror(self._t("input_error_title"), self._t("port_must_be_number"))
            return
        if not broker or not topic:
            messagebox.showerror(self._t("input_error_title"), self._t("broker_topic_required"))
            return

        try:
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        self.mqtt_client.user_data_set({"topic": topic})

        try:
            self.mqtt_client.connect_async(broker, port, 60)
            self.mqtt_client.loop_start()
            self._set_status("mqtt_connecting", COLOR_INFO, broker=broker, port=port)
        except Exception as exc:
            self._set_status("mqtt_connect_failed", COLOR_ERROR, error=exc)
            messagebox.showerror(self._t("mqtt_connect_error_title"), str(exc))

    def on_mqtt_connect(self, client: Any, userdata: Any, flags: Any, rc: int, properties: Any = None) -> None:
        if rc == 0:
            topic = userdata.get("topic", "vehicle/gps_data") if isinstance(userdata, dict) else "vehicle/gps_data"
            client.subscribe(topic)
            self.is_mqtt_connected = True
            self.data_buffer.clear()
            self._update_buffer_progress()
            self.connect_button.config(state=tk.DISABLED)
            self.disconnect_button.config(state=tk.NORMAL)
            self._set_connection_state(True)
            self._set_status("mqtt_connected", COLOR_SUCCESS, topic=topic)
        else:
            self._set_status("mqtt_connect_failed_code", COLOR_ERROR, code=rc)

    def on_mqtt_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            self.root.after(0, self.process_data_point, data)
        except Exception as exc:
            self.root.after(0, self._set_status, "mqtt_parse_failed", COLOR_ERROR, error=exc)

    def on_mqtt_disconnect(self, client: Any, userdata: Any, rc: int, properties: Any = None) -> None:
        self.is_mqtt_connected = False
        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self._set_connection_state(False)
        self._set_status("mqtt_disconnected", COLOR_WARNING)

    def disconnect_mqtt(self) -> None:
        if self.mqtt_client is None:
            return
        try:
            self.mqtt_client.loop_stop()
            if self.mqtt_client.is_connected():
                self.mqtt_client.disconnect()
        except Exception:
            pass
        self.is_mqtt_connected = False
        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self._set_connection_state(False)

    def process_data_point(self, raw_data: dict[str, Any]) -> None:
        if self.model is None or self.scaler is None:
            self._set_status("model_not_loaded", COLOR_ERROR)
            return
        if not isinstance(raw_data, dict):
            self._set_status("invalid_input_data", COLOR_ERROR)
            return

        feature_vector, warnings = build_feature_vector(raw_data, self.language)
        if feature_vector is None:
            missing = [col for col in FEATURE_COLS if col not in raw_data]
            if missing:
                self._set_status("missing_fields", COLOR_ERROR, fields=", ".join(missing))
            else:
                self._set_status(warnings[0] if warnings else "invalid_input_data", COLOR_ERROR)
            self._update_detail_panel(raw_data, warnings, None, None, None)
            return

        self.data_buffer.append(feature_vector)
        self._update_buffer_progress()

        latitude = parse_float(raw_data.get("Lat"))
        longitude = parse_float(raw_data.get("Lng"))
        gps_speed = parse_float(raw_data.get("gps速度"))
        self._request_speed_limit_if_needed(latitude, longitude)

        prediction: tuple[int, float] | None = None
        warning_source = ""
        if len(self.data_buffer) == WINDOW_SIZE:
            try:
                prediction = predict_behavior(self.model, self.scaler, list(self.data_buffer))
            except Exception as exc:
                self._set_status("prediction_failed", COLOR_ERROR, error=exc)
                self._update_detail_panel(raw_data, warnings, None, None, None)
                return

        api_overspeed = self._is_api_overspeed(gps_speed)
        self.last_prediction_state = (prediction, api_overspeed)
        display_label, confidence, color = self._build_prediction_display(prediction, api_overspeed)
        self.prediction_label.config(text=display_label, foreground=color)

        if prediction is not None and prediction[0] != 0 and prediction[1] >= CONFIDENCE_THRESHOLD:
            warning_source = "model"
        if api_overspeed:
            warning_source = "api" if not warning_source else "model+api"

        if warning_source:
            self._log_warning_event(raw_data, prediction, warning_source)

        self._update_speed_limit_label(gps_speed, api_overspeed)
        self._update_detail_panel(raw_data, warnings, prediction, api_overspeed, warning_source)

    def _build_prediction_display(
        self,
        prediction: tuple[int, float] | None,
        api_overspeed: bool,
    ) -> tuple[str, float | None, str]:
        if api_overspeed and prediction is None:
            return (
                self._t("prediction_api_wait", current=len(self.data_buffer), total=WINDOW_SIZE),
                None,
                COLOR_ERROR,
            )
        if prediction is None:
            return (
                self._t("prediction_wait", current=len(self.data_buffer), total=WINDOW_SIZE),
                None,
                COLOR_MUTED,
            )

        predicted_index, confidence = prediction
        label = self._class_label(predicted_index)
        if api_overspeed:
            return self._t("prediction_api_plus", label=label, confidence=confidence), confidence, COLOR_ERROR
        if confidence < CONFIDENCE_THRESHOLD:
            return self._t("prediction_uncertain", label=label, confidence=confidence), confidence, COLOR_WARNING
        if predicted_index == 0:
            return self._t("prediction_result", label=label, confidence=confidence), confidence, COLOR_SUCCESS
        return self._t("prediction_result", label=label, confidence=confidence), confidence, COLOR_ERROR

    def _request_speed_limit_if_needed(self, latitude: float | None, longitude: float | None) -> None:
        if not self.amap_api_key or latitude is None or longitude is None:
            return
        if self.api_request_running:
            return
        if time.time() - self.last_api_call_time < API_CALL_INTERVAL_SECONDS:
            return

        self.api_request_running = True
        self.last_api_call_time = time.time()
        thread = threading.Thread(
            target=self._speed_limit_worker,
            args=(latitude, longitude),
            daemon=True,
        )
        thread.start()

    def _speed_limit_worker(self, latitude: float, longitude: float) -> None:
        try:
            speed_limit = get_speed_limit_from_amap(self.amap_api_key, latitude, longitude)
            self.api_queue.put(("ok", speed_limit, None))
        except Exception as exc:
            self.api_queue.put(("error", None, str(exc)))

    def _poll_api_queue(self) -> None:
        try:
            while True:
                status, speed_limit, error = self.api_queue.get_nowait()
                self.api_request_running = False
                if status == "ok":
                    self.current_road_speed_limit = speed_limit
                    self.last_api_error = None
                else:
                    self.last_api_error = error
        except Empty:
            pass
        self.root.after(300, self._poll_api_queue)

    def _is_api_overspeed(self, gps_speed: float | None) -> bool:
        if gps_speed is None or self.current_road_speed_limit is None:
            return False
        return gps_speed > self.current_road_speed_limit * 1.1

    def _update_speed_limit_label(self, gps_speed: float | None, api_overspeed: bool, remember: bool = True) -> None:
        if remember:
            self.last_speed_state = (gps_speed, api_overspeed)
        if not self.amap_api_key:
            self.road_speed_value.config(text=self._t("road_speed_limit_value"), fg=COLOR_MUTED, bg=COLOR_SURFACE)
            return
        if self.last_api_error:
            self.road_speed_value.config(text=self._t("speed_api_failed", error=self.last_api_error), fg=COLOR_WARNING, bg=COLOR_ORANGE_SOFT)
            return
        if self.current_road_speed_limit is None:
            self.road_speed_value.config(text=self._t("speed_pending"), fg=COLOR_MUTED, bg=COLOR_SURFACE_ALT)
            return

        speed_text = "-" if gps_speed is None else f"{gps_speed:.1f} km/h"
        color = COLOR_ERROR if api_overspeed else COLOR_INFO
        bg = COLOR_RED_SOFT if api_overspeed else COLOR_BLUE_SOFT
        self.road_speed_value.config(text=f"{self.current_road_speed_limit} km/h · {speed_text}", fg=color, bg=bg)

    def _update_detail_panel(
        self,
        raw_data: dict[str, Any],
        warnings: list[str],
        prediction: tuple[int, float] | None,
        api_overspeed: bool | None,
        warning_source: str | None,
        remember: bool = True,
    ) -> None:
        if remember:
            self.last_detail_state = (raw_data, warnings, prediction, api_overspeed, warning_source)
        lines = [
            self._t("latest_data_point"),
            "-" * 40,
            f"{self._t('buffer')}: {len(self.data_buffer)}/{WINDOW_SIZE}",
            f"{self._t('vehicle_id')}: {raw_data.get('vid_md5', '-')}",
            f"{self._t('gps_time')}: {raw_data.get('gps时间', raw_data.get('时间戳', '-'))}",
            f"Lng: {raw_data.get('Lng', '-')}",
            f"Lat: {raw_data.get('Lat', '-')}",
            f"{self._t('gps_speed')}: {raw_data.get('gps速度', '-')}",
            f"{self._t('vss_speed')}: {raw_data.get('vss速度', '-')}",
            f"{self._t('acc_status')}: {raw_data.get('ACC状态', '-')}",
            f"{self._t('road_speed_limit')}: {self.current_road_speed_limit if self.current_road_speed_limit is not None else '-'}",
        ]
        if prediction is not None:
            predicted_index, confidence = prediction
            lines.append(self._t("model_result", label=self._class_label(predicted_index), confidence=confidence))
        if api_overspeed:
            lines.append(self._t("api_overspeed_detail"))
        if warning_source:
            lines.append(f"{self._t('warning_source')}: {warning_source}")
        if warnings:
            lines.extend(["", self._t("data_notes")])
            lines.extend(f"- {warning}" for warning in warnings)
        self._write_detail("\n".join(lines))

    def _log_warning_event(
        self,
        raw_data: dict[str, Any],
        prediction: tuple[int, float] | None,
        warning_source: str,
    ) -> None:
        predicted_class = ""
        confidence = ""
        if prediction is not None:
            predicted_index, confidence_value = prediction
            predicted_class = self._class_label(predicted_index)
            confidence = f"{confidence_value:.4f}"

        append_event_log(
            {
                "timestamp": raw_data.get("gps时间", raw_data.get("时间戳", time.strftime("%Y-%m-%d %H:%M:%S"))),
                "vehicle_id": raw_data.get("vid_md5", ""),
                "longitude": raw_data.get("Lng", ""),
                "latitude": raw_data.get("Lat", ""),
                "gps_speed": raw_data.get("gps速度", ""),
                "road_speed_limit": self.current_road_speed_limit or "",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "warning_source": warning_source,
            }
        )

    def on_closing(self) -> None:
        self.stop_simulation_event.set()
        self.disconnect_mqtt()
        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()
    App(app_root)
    app_root.mainloop()
