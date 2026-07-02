from __future__ import annotations

import csv
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import requests
import torch

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

TRANSLATIONS = {
    "zh": {
        "loading_model": "正在加载模型...",
        "model_loaded": "模型和 scaler 加载成功",
        "model_load_failed": "模型加载失败: {error}",
        "api_enabled": "地图 API 已启用",
        "api_disabled": "未配置 AMAP_API_KEY，地图限速检查已关闭",
        "model_not_loaded": "模型未加载，无法预测",
        "invalid_input_data": "输入数据格式无效",
        "missing_fields": "缺少模型字段: {fields}",
        "invalid_field": "字段 {field} 不是有效数字，已使用 0",
        "prediction_wait": "等待窗口填满 ({current}/{total})",
        "prediction_api_wait": "API 超速，等待模型窗口 ({current}/{total})",
        "prediction_api_plus": "API 超速 + {label}",
        "prediction_uncertain": "不确定: {label}",
        "prediction_result": "{label}",
        "prediction_failed": "预测失败: {error}",
        "mqtt_unavailable": "请先安装 paho-mqtt",
        "mqtt_already_connected": "MQTT 已连接",
        "port_must_be_number": "端口必须是数字",
        "broker_topic_required": "Broker 地址和 Topic 不能为空",
        "mqtt_connecting": "正在连接 MQTT: {broker}:{port}",
        "mqtt_connected": "MQTT 已连接，订阅 Topic: {topic}",
        "mqtt_connect_failed": "MQTT 连接失败: {error}",
        "mqtt_connect_failed_code": "MQTT 连接失败，返回码: {code}",
        "mqtt_parse_failed": "MQTT 数据解析失败: {error}",
        "mqtt_disconnected": "MQTT 已断开",
        "csv_selected": "已选择 CSV: {filename}",
        "csv_missing": "CSV 文件不存在",
        "select_csv_first": "请先选择 CSV 文件",
        "simulation_already_running": "CSV 模拟已经在运行",
        "csv_simulating": "CSV 模拟进行中",
        "csv_failed": "CSV 模拟失败: {error}",
        "csv_stopped": "CSV 模拟已停止",
        "csv_completed": "CSV 模拟完成",
        "speed_no_api_key": "未配置 AMAP_API_KEY",
        "speed_pending": "正在获取或暂无结果",
        "speed_api_failed": "API 获取失败: {error}",
        "speed_current": "{limit} km/h，当前速度 {speed}",
        "api_overspeed_detail": "API 判断: 当前速度超过道路限速 10%",
        "warning_source": "预警来源",
    },
    "en": {
        "loading_model": "Loading model...",
        "model_loaded": "Model and scaler loaded",
        "model_load_failed": "Model load failed: {error}",
        "api_enabled": "Map API enabled",
        "api_disabled": "AMAP_API_KEY is not set. Map speed-limit checks are off",
        "model_not_loaded": "Model is not loaded. Prediction is unavailable",
        "invalid_input_data": "Input data format is invalid",
        "missing_fields": "Missing model fields: {fields}",
        "invalid_field": "Field {field} is not a valid number. Used 0 instead",
        "prediction_wait": "Waiting for model window ({current}/{total})",
        "prediction_api_wait": "API overspeed. Waiting for model window ({current}/{total})",
        "prediction_api_plus": "API overspeed + {label}",
        "prediction_uncertain": "Uncertain: {label}",
        "prediction_result": "{label}",
        "prediction_failed": "Prediction failed: {error}",
        "mqtt_unavailable": "Please install paho-mqtt first",
        "mqtt_already_connected": "MQTT is already connected",
        "port_must_be_number": "Port must be a number",
        "broker_topic_required": "Broker address and Topic cannot be empty",
        "mqtt_connecting": "Connecting to MQTT: {broker}:{port}",
        "mqtt_connected": "MQTT connected. Subscribed to Topic: {topic}",
        "mqtt_connect_failed": "MQTT connection failed: {error}",
        "mqtt_connect_failed_code": "MQTT connection failed. Return code: {code}",
        "mqtt_parse_failed": "Failed to parse MQTT data: {error}",
        "mqtt_disconnected": "MQTT disconnected",
        "csv_selected": "Selected CSV: {filename}",
        "csv_missing": "CSV file does not exist",
        "select_csv_first": "Please select a CSV file first",
        "simulation_already_running": "CSV simulation is already running",
        "csv_simulating": "CSV simulation is running",
        "csv_failed": "CSV simulation failed: {error}",
        "csv_stopped": "CSV simulation stopped",
        "csv_completed": "CSV simulation completed",
        "speed_no_api_key": "AMAP_API_KEY is not set",
        "speed_pending": "Fetching or no result yet",
        "speed_api_failed": "API request failed: {error}",
        "speed_current": "{limit} km/h, current speed {speed}",
        "api_overspeed_detail": "API check: current speed is more than 10% over the road speed limit",
        "warning_source": "Warning source",
    },
}


def load_model_and_scaler() -> tuple[CNNLSTMModel, Any]:
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
    messages = TRANSLATIONS.get(language, TRANSLATIONS["zh"])
    missing = [col for col in FEATURE_COLS if col not in raw_data]
    if missing:
        return None, [messages["missing_fields"].format(fields=", ".join(missing))]

    values: list[float] = []
    warnings: list[str] = []
    for col in FEATURE_COLS:
        if "ACC" in col:
            values.append(parse_acc_status(raw_data.get(col)))
            continue

        parsed = parse_float(raw_data.get(col))
        if parsed is None:
            warnings.append(messages["invalid_field"].format(field=col))
            parsed = 0.0
        values.append(parsed)
    return values, warnings


def predict_behavior(model: CNNLSTMModel, scaler: Any, window_values: list[list[float]]) -> tuple[int, float]:
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


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


class BackendRuntime:
    def __init__(self, emit: Callable[[dict[str, Any]], None]):
        self.emit = emit
        self.language = "en"
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
        self.api_request_running = False
        self.last_api_call_time = 0.0
        self.current_road_speed_limit: int | None = None
        self.last_api_error: str | None = None
        self.last_raw_data: dict[str, Any] | None = None
        self.last_prediction: dict[str, Any] = self._empty_prediction()
        self.last_status = self._t("loading_model")
        self.last_status_kind = "loading"

    def start(self) -> None:
        self.emit_state()
        self._load_runtime_assets()

    def _t(self, key: str, **params: Any) -> str:
        template = TRANSLATIONS.get(self.language, TRANSLATIONS["zh"]).get(key, key)
        return template.format(**params) if params else template

    def _class_label(self, index: int) -> str:
        return CLASS_LABELS.get(self.language, CLASS_LABELS["zh"]).get(index, str(index))

    def _emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.emit({"event": event, "payload": to_jsonable(payload or {})})

    def _set_status(self, key: str, kind: str = "info", **params: Any) -> None:
        self.last_status = self._t(key, **params)
        self.last_status_kind = kind
        self._emit("status_changed", {"message": self.last_status, "kind": kind})
        self.emit_state()

    def _load_runtime_assets(self) -> None:
        try:
            self.model, self.scaler = load_model_and_scaler()
            api_status = self._t("api_enabled") if self.amap_api_key else self._t("api_disabled")
            self._set_status("model_loaded", "success")
            self._emit("model_api_status_changed", self.model_api_status(api_status))
        except Exception as exc:
            self.model = None
            self.scaler = None
            self._set_status("model_load_failed", "error", error=str(exc))
            self._emit("model_api_status_changed", self.model_api_status(str(exc), model_ok=False))

    def model_api_status(self, api_text: str | None = None, model_ok: bool | None = None) -> dict[str, Any]:
        return {
            "modelLoaded": self.model is not None if model_ok is None else model_ok,
            "modelFile": MODEL_PATH.name,
            "apiEnabled": bool(self.amap_api_key),
            "apiText": api_text or (self._t("api_enabled") if self.amap_api_key else self._t("api_disabled")),
            "apiUrl": AMAP_REVERSE_GEOCODE_URL if self.amap_api_key else "",
        }

    def _empty_prediction(self) -> dict[str, Any]:
        return {
            "label": "-",
            "message": "-",
            "confidence": None,
            "classIndex": None,
            "kind": "muted",
            "warningSource": "",
            "apiOverspeed": False,
        }

    def emit_state(self) -> None:
        self._emit(
            "state_snapshot",
            {
                "language": self.language,
                "status": {"message": self.last_status, "kind": self.last_status_kind},
                "mqtt": {"connected": self.is_mqtt_connected},
                "csv": {
                    "selectedPath": self.csv_file_path,
                    "selectedName": Path(self.csv_file_path).name if self.csv_file_path else "",
                    "simulating": self.is_simulating,
                },
                "window": {"current": len(self.data_buffer), "total": WINDOW_SIZE},
                "prediction": self.last_prediction,
                "speedLimit": self.speed_limit_payload(None, False),
                "latestData": self.last_raw_data or {},
                "modelApi": self.model_api_status(),
            },
        )

    def handle_command(self, command: dict[str, Any]) -> None:
        name = command.get("command")
        payload = command.get("payload") or {}
        try:
            if name == "get_initial_state":
                self.emit_state()
            elif name == "set_language":
                self.set_language(str(payload.get("language", "zh")))
            elif name == "connect_mqtt":
                self.connect_mqtt(payload)
            elif name == "disconnect_mqtt":
                self.disconnect_mqtt()
            elif name == "select_csv":
                self.select_csv(str(payload.get("path", "")))
            elif name == "start_csv_simulation":
                self.start_csv_simulation()
            elif name == "stop_csv_simulation":
                self.stop_csv_simulation()
            elif name == "reset_replay":
                self.reset_replay_state()
            elif name == "shutdown":
                self.shutdown()
            else:
                self._emit("error", {"message": f"Unknown command: {name}"})
        except Exception as exc:
            self._emit("error", {"message": str(exc)})

    def set_language(self, language: str) -> None:
        self.language = "en" if language == "en" else "zh"
        self._emit("language_changed", {"language": self.language})
        self.emit_state()

    def select_csv(self, path: str) -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            self._set_status("csv_missing", "error")
            return
        self.csv_file_path = str(csv_path)
        self._set_status("csv_selected", "info", filename=csv_path.name)

    def start_csv_simulation(self) -> None:
        if self.is_simulating:
            self._set_status("simulation_already_running", "info")
            return
        if not self.csv_file_path:
            self._set_status("select_csv_first", "warning")
            return

        self.data_buffer.clear()
        self.stop_simulation_event.clear()
        self.is_simulating = True
        self._set_status("csv_simulating", "info")
        self.simulation_thread = threading.Thread(target=self._simulation_worker, daemon=True)
        self.simulation_thread.start()

    def _simulation_worker(self) -> None:
        try:
            assert self.csv_file_path is not None
            data = pd.read_csv(self.csv_file_path)
            for _, row in data.iterrows():
                if self.stop_simulation_event.is_set():
                    break
                self.process_data_point(row.to_dict())
                time.sleep(self.simulation_delay)
        except Exception as exc:
            self._set_status("csv_failed", "error", error=str(exc))
        finally:
            self.is_simulating = False
            if self.stop_simulation_event.is_set():
                self._set_status("csv_stopped", "warning")
            else:
                self._set_status("csv_completed", "success")

    def stop_csv_simulation(self) -> None:
        self.stop_simulation_event.set()

    def reset_replay_state(self) -> None:
        self.stop_simulation_event.set()
        self.data_buffer.clear()
        self.current_road_speed_limit = None
        self.last_api_error = None
        self.last_raw_data = None
        self.last_prediction = self._empty_prediction()
        self._emit("warnings_reset", {})
        self._set_status("csv_stopped", "muted")
        self.emit_state()

    def connect_mqtt(self, payload: dict[str, Any]) -> None:
        if mqtt is None:
            self._set_status("mqtt_unavailable", "error")
            return
        if self.is_mqtt_connected:
            self._set_status("mqtt_already_connected", "info")
            return

        broker = str(payload.get("broker", "")).strip()
        topic = str(payload.get("topic", "")).strip()
        try:
            port = int(str(payload.get("port", "1883")).strip())
        except ValueError:
            self._set_status("port_must_be_number", "error")
            return
        if not broker or not topic:
            self._set_status("broker_topic_required", "error")
            return

        try:
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.user_data_set({"topic": topic})

        try:
            self.mqtt_client.connect_async(broker, port, 60)
            self.mqtt_client.loop_start()
            self._set_status("mqtt_connecting", "info", broker=broker, port=port)
        except Exception as exc:
            self._set_status("mqtt_connect_failed", "error", error=str(exc))

    def _on_mqtt_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        try:
            code = int(rc)
        except Exception:
            code = 0 if str(rc).lower() == "success" else -1
        if code == 0:
            topic = userdata.get("topic", "vehicle/gps_data") if isinstance(userdata, dict) else "vehicle/gps_data"
            client.subscribe(topic)
            self.is_mqtt_connected = True
            self.data_buffer.clear()
            self._set_status("mqtt_connected", "success", topic=topic)
            self._emit("mqtt_changed", {"connected": True, "topic": topic})
        else:
            self._set_status("mqtt_connect_failed_code", "error", code=code)

    def _on_mqtt_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            self.process_data_point(data)
        except Exception as exc:
            self._set_status("mqtt_parse_failed", "error", error=str(exc))

    def _on_mqtt_disconnect(self, *args: Any) -> None:
        self.is_mqtt_connected = False
        self._set_status("mqtt_disconnected", "warning")
        self._emit("mqtt_changed", {"connected": False})

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
        self._emit("mqtt_changed", {"connected": False})
        self.emit_state()

    def process_data_point(self, raw_data: dict[str, Any]) -> None:
        if self.model is None or self.scaler is None:
            self._set_status("model_not_loaded", "error")
            return
        if not isinstance(raw_data, dict):
            self._set_status("invalid_input_data", "error")
            return

        self.last_raw_data = to_jsonable(raw_data)
        feature_vector, warnings = build_feature_vector(raw_data, self.language)
        if feature_vector is None:
            missing = [col for col in FEATURE_COLS if col not in raw_data]
            if missing:
                self._set_status("missing_fields", "error", fields=", ".join(missing))
            elif warnings:
                self._emit("error", {"message": warnings[0]})
            self._emit_latest(raw_data, warnings, None, False, "")
            return

        self.data_buffer.append(feature_vector)
        latitude = parse_float(raw_data.get("Lat"))
        longitude = parse_float(raw_data.get("Lng"))
        gps_speed = parse_float(raw_data.get("gps速度", raw_data.get("gps閫熷害")))
        self._request_speed_limit_if_needed(latitude, longitude)

        prediction: tuple[int, float] | None = None
        if len(self.data_buffer) == WINDOW_SIZE:
            try:
                prediction = predict_behavior(self.model, self.scaler, list(self.data_buffer))
            except Exception as exc:
                self._set_status("prediction_failed", "error", error=str(exc))
                self._emit_latest(raw_data, warnings, None, False, "")
                return

        api_overspeed = self._is_api_overspeed(gps_speed)
        warning_source = ""
        if prediction is not None and prediction[0] != 0 and prediction[1] >= CONFIDENCE_THRESHOLD:
            warning_source = "model"
        if api_overspeed:
            warning_source = "api" if not warning_source else "model+api"

        self.last_prediction = self.prediction_payload(prediction, api_overspeed, warning_source)
        if warning_source:
            self._log_warning_event(raw_data, prediction, warning_source)

        self._emit("prediction_changed", self.last_prediction)
        self._emit("speed_limit_changed", self.speed_limit_payload(gps_speed, api_overspeed))
        self._emit_latest(raw_data, warnings, prediction, api_overspeed, warning_source)
        self.emit_state()

    def prediction_payload(
        self,
        prediction: tuple[int, float] | None,
        api_overspeed: bool,
        warning_source: str,
    ) -> dict[str, Any]:
        if api_overspeed and prediction is None:
            return {
                "label": self._t("prediction_api_wait", current=len(self.data_buffer), total=WINDOW_SIZE),
                "message": self._t("prediction_api_wait", current=len(self.data_buffer), total=WINDOW_SIZE),
                "confidence": None,
                "classIndex": None,
                "kind": "danger",
                "warningSource": warning_source,
                "apiOverspeed": True,
            }
        if prediction is None:
            message = self._t("prediction_wait", current=len(self.data_buffer), total=WINDOW_SIZE)
            return {
                "label": message,
                "message": message,
                "confidence": None,
                "classIndex": None,
                "kind": "muted",
                "warningSource": warning_source,
                "apiOverspeed": api_overspeed,
            }

        predicted_index, confidence = prediction
        label = self._class_label(predicted_index)
        if api_overspeed:
            message = self._t("prediction_api_plus", label=label)
            kind = "danger"
        elif confidence < CONFIDENCE_THRESHOLD:
            message = self._t("prediction_uncertain", label=label)
            kind = "warning"
        elif predicted_index == 0:
            message = self._t("prediction_result", label=label)
            kind = "success"
        else:
            message = self._t("prediction_result", label=label)
            kind = "danger"
        return {
            "label": label,
            "message": message,
            "confidence": confidence,
            "classIndex": predicted_index,
            "kind": kind,
            "warningSource": warning_source,
            "apiOverspeed": api_overspeed,
        }

    def _request_speed_limit_if_needed(self, latitude: float | None, longitude: float | None) -> None:
        if not self.amap_api_key or latitude is None or longitude is None:
            return
        if self.api_request_running:
            return
        if time.time() - self.last_api_call_time < API_CALL_INTERVAL_SECONDS:
            return
        self.api_request_running = True
        self.last_api_call_time = time.time()
        threading.Thread(target=self._speed_limit_worker, args=(latitude, longitude), daemon=True).start()

    def _speed_limit_worker(self, latitude: float, longitude: float) -> None:
        try:
            self.current_road_speed_limit = get_speed_limit_from_amap(self.amap_api_key, latitude, longitude)
            self.last_api_error = None
        except Exception as exc:
            self.last_api_error = str(exc)
        finally:
            self.api_request_running = False
            self._emit("speed_limit_changed", self.speed_limit_payload(None, False))

    def _is_api_overspeed(self, gps_speed: float | None) -> bool:
        if gps_speed is None or self.current_road_speed_limit is None:
            return False
        return gps_speed > self.current_road_speed_limit * 1.1

    def speed_limit_payload(self, gps_speed: float | None, api_overspeed: bool) -> dict[str, Any]:
        if not self.amap_api_key:
            return {"limit": None, "gpsSpeed": gps_speed, "message": self._t("speed_no_api_key"), "kind": "muted"}
        if self.last_api_error:
            return {
                "limit": self.current_road_speed_limit,
                "gpsSpeed": gps_speed,
                "message": self._t("speed_api_failed", error=self.last_api_error),
                "kind": "warning",
            }
        if self.current_road_speed_limit is None:
            return {"limit": None, "gpsSpeed": gps_speed, "message": self._t("speed_pending"), "kind": "muted"}
        speed_text = "-" if gps_speed is None else f"{gps_speed:.1f} km/h"
        return {
            "limit": self.current_road_speed_limit,
            "gpsSpeed": gps_speed,
            "message": self._t("speed_current", limit=self.current_road_speed_limit, speed=speed_text),
            "kind": "danger" if api_overspeed else "info",
        }

    def _emit_latest(
        self,
        raw_data: dict[str, Any],
        warnings: list[str],
        prediction: tuple[int, float] | None,
        api_overspeed: bool,
        warning_source: str,
    ) -> None:
        self._emit(
            "latest_data_changed",
            {
                "raw": raw_data,
                "warnings": warnings,
                "window": {"current": len(self.data_buffer), "total": WINDOW_SIZE},
                "prediction": self.prediction_payload(prediction, api_overspeed, warning_source),
                "apiOverspeed": api_overspeed,
                "warningSource": warning_source,
            },
        )

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

        row = {
            "timestamp": raw_data.get("gps时间", raw_data.get("gps鏃堕棿", time.strftime("%Y-%m-%d %H:%M:%S"))),
            "vehicle_id": raw_data.get("vid_md5", ""),
            "longitude": raw_data.get("Lng", ""),
            "latitude": raw_data.get("Lat", ""),
            "gps_speed": raw_data.get("gps速度", raw_data.get("gps閫熷害", "")),
            "road_speed_limit": self.current_road_speed_limit or "",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "warning_source": warning_source,
        }
        append_event_log(row)
        self._emit("warning_logged", row)

    def shutdown(self) -> None:
        self.stop_simulation_event.set()
        self.disconnect_mqtt()
