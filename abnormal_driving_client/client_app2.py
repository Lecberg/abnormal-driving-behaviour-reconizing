import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import torch
import joblib
from collections import deque
import time
import threading
import json
import requests
import paho.mqtt.client as mqtt

from CNNLSTMModel import CNNLSTMModel, FEATURE_COLS, WINDOW_SIZE, DEVICE

# --- 全局常量 ---
MODEL_PATH = "abnormal_driving_client/best_model.pth"
SCALER_PATH = "abnormal_driving_client/scaler.gz"
NUM_FEATURES = len(FEATURE_COLS)
NUM_CLASSES = 4
CLASS_LABELS = {
    0: "正常驾驶",
    1: "高速行驶",
    2: "急加速/急减速",
    3: "曲折行驶"
}
ABNORMAL_CLASSES = [1, 2, 3]

MAP_API_PROVIDER = "amap"
AMAP_API_KEY = "YOUR_AMAP_WEB_SERVICE_API_KEY" # <--- 替换成高德Web服务API Key
AMAP_REVERSE_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"

COLOR_PRIMARY = "#2E3F4F"
COLOR_SECONDARY = "#4A6F8A"
COLOR_ACCENT = "#F7B733"
COLOR_TEXT = "#FFFFFF"
COLOR_SUCCESS = "#2ECC71"
COLOR_ERROR = "#E74C3C"
COLOR_WARNING = "#F39C12"
COLOR_INFO = "#3498DB"
COLOR_OVERSPEED = "#FF00FF"

FONT_FAMILY_CN = "微软雅黑"
FONT_NORMAL = (FONT_FAMILY_CN, 10)
FONT_BOLD = (FONT_FAMILY_CN, 12, "bold")
FONT_LARGE_BOLD = (FONT_FAMILY_CN, 16, "bold")
FONT_STATUS = (FONT_FAMILY_CN, 11)
FONT_PREDICTION = (FONT_FAMILY_CN, 18, "bold")
FONT_TEXT_AREA = ("Consolas", 10)
FONT_OVERSPEED_INFO = (FONT_FAMILY_CN, 10, "italic")

model = None
scaler = None
data_buffer = deque(maxlen=WINDOW_SIZE)
mqtt_client = None
is_connected_to_mqtt = False
simulation_thread = None
stop_simulation_event = threading.Event()
is_simulating = False
last_api_call_time = 0
API_CALL_INTERVAL = 5

def load_model_and_scaler():
    global model, scaler
    try:
        model = CNNLSTMModel(input_size=NUM_FEATURES, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("模型加载成功!")
        scaler = joblib.load(SCALER_PATH)
        print("Scaler加载成功!")
        return True
    except FileNotFoundError:
        messagebox.showerror("加载错误", f"模型或Scaler文件未找到。\n请检查路径:\n{MODEL_PATH}\n{SCALER_PATH}")
        return False
    except Exception as e:
        messagebox.showerror("加载错误", f"加载模型或Scaler失败: {e}")
        return False

def predict_behavior(window_features_np):
    if model is None or scaler is None: return -1
    try:
        features_tensor = torch.FloatTensor(window_features_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(features_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.item()
    except Exception as e:
        print(f"预测出错: {e}")
        return -1

def get_speed_limit_from_api(latitude, longitude):
    global last_api_call_time
    current_time = time.time()
    if current_time - last_api_call_time < API_CALL_INTERVAL:
        return None

    last_api_call_time = current_time
    if MAP_API_PROVIDER == "amap":
        params = {"key": AMAP_API_KEY, "location": f"{longitude},{latitude}", "extensions": "base", "radius": "1000", "roadlevel": "1"}
        try:
            response = requests.get(AMAP_REVERSE_GEOCODE_URL, params=params, timeout=3)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "1" and "regeocode" in data:
                regeocode_data = data["regeocode"]
                roads = regeocode_data.get("roads")
                if roads and len(roads) > 0:
                    speed_limit_str = roads[0].get("speed")
                    if speed_limit_str and speed_limit_str.isdigit(): return int(speed_limit_str)
                    elif speed_limit_str == "[]": return None
                roadinters = regeocode_data.get("roadinters")
                if roadinters and len(roadinters) > 0:
                    speed_limit_str = roadinters[0].get("speed")
                    if speed_limit_str and speed_limit_str.isdigit(): return int(speed_limit_str)
            return None
        except requests.exceptions.RequestException as e: print(f"高德地图API请求失败: {e}"); return None
        except (json.JSONDecodeError, KeyError, IndexError) as e: print(f"解析高德地图API响应失败: {e}"); return None
    else: print(f"未知的地图API提供商: {MAP_API_PROVIDER}"); return None

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("实时异常驾驶行为检测客户端 (模拟+API限速)")
        self.root.geometry("750x780")
        self.root.configure(bg=COLOR_PRIMARY)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", background=COLOR_PRIMARY, foreground=COLOR_TEXT, font=FONT_NORMAL, padding=5)
        self.style.configure("Title.TLabel", font=FONT_LARGE_BOLD, foreground=COLOR_ACCENT)
        self.style.configure("Status.TLabel", font=FONT_STATUS, padding=10)
        self.style.configure("Prediction.TLabel", font=FONT_PREDICTION, padding=10)
        self.style.configure("Overspeed.TLabel", font=FONT_OVERSPEED_INFO, padding=5, foreground=COLOR_OVERSPEED)
        self.style.configure("TEntry", fieldbackground="#FFFFFF", foreground=COLOR_PRIMARY, font=FONT_NORMAL, padding=5)
        self.style.map("TEntry", foreground=[('focus', COLOR_PRIMARY)])
        self.style.configure("TButton", font=FONT_BOLD, padding=(10, 5))
        self.style.map("TButton", foreground=[('active', COLOR_PRIMARY), ('!disabled', COLOR_TEXT)], background=[('active', COLOR_ACCENT), ('!disabled', COLOR_SECONDARY)], relief=[('pressed', 'sunken'), ('!pressed', 'raised')])
        self.style.configure("Disconnect.TButton", background=COLOR_WARNING)
        self.style.map("Disconnect.TButton", background=[('active', COLOR_ERROR)])
        self.style.configure("Simulate.TButton", background=COLOR_INFO)
        self.style.map("Simulate.TButton", background=[('active', COLOR_ACCENT)])
        self.style.configure("TLabelframe", background=COLOR_PRIMARY, relief="groove", borderwidth=2)
        self.style.configure("TLabelframe.Label", background=COLOR_PRIMARY, foreground=COLOR_ACCENT, font=FONT_BOLD)
        self.style.configure("Custom.TFrame", background=COLOR_PRIMARY)

        title_label = ttk.Label(root, text="异常驾驶行为实时监测系统", style="Title.TLabel")
        title_label.pack(pady=(20,10), side=tk.TOP)

        self.mqtt_frame = ttk.LabelFrame(root, text="MQTT 连接设置", style="TLabelframe")
        self.mqtt_frame.pack(padx=20, pady=10, fill="x", ipady=10, side=tk.TOP)
        self.mqtt_frame.columnconfigure(1, weight=1)
        ttk.Label(self.mqtt_frame, text="Broker 地址:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.broker_entry = ttk.Entry(self.mqtt_frame, width=35); self.broker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew"); self.broker_entry.insert(0, "localhost")
        ttk.Label(self.mqtt_frame, text="端口:").grid(row=0, column=2, padx=(10,5), pady=5, sticky="w")
        self.port_entry = ttk.Entry(self.mqtt_frame, width=10); self.port_entry.grid(row=0, column=3, padx=(0,10), pady=5, sticky="w"); self.port_entry.insert(0, "1883")
        ttk.Label(self.mqtt_frame, text="Topic:").grid(row=1, column=0, padx=(10,5), pady=5, sticky="w")
        self.topic_entry = ttk.Entry(self.mqtt_frame, width=35); self.topic_entry.grid(row=1, column=1, columnspan=3, padx=(5,10), pady=5, sticky="ew"); self.topic_entry.insert(0, "vehicle/gps_data")

        status_prediction_frame = ttk.Frame(root, style="Custom.TFrame", padding=10)
        status_prediction_frame.pack(pady=5, fill="x", padx=20, side=tk.TOP)
        self.status_label = ttk.Label(status_prediction_frame, text="状态: 未连接", style="Status.TLabel", anchor="center"); self.status_label.pack(pady=(0,5), fill="x")
        self.prediction_label = ttk.Label(status_prediction_frame, text="预测行为: -", style="Prediction.TLabel", anchor="center"); self.prediction_label.pack(pady=5, fill="x")
        self.overspeed_label = ttk.Label(status_prediction_frame, text="", style="Overspeed.TLabel", anchor="center"); self.overspeed_label.pack(pady=(0,5), fill="x")

        button_frame = ttk.Frame(root, style="Custom.TFrame")
        button_frame.pack(side=tk.BOTTOM, pady=(5, 10), fill="x")
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        self.connect_button = ttk.Button(button_frame, text="连接到 MQTT", command=self.connect_mqtt, style="TButton"); self.connect_button.grid(row=0, column=0, padx=20, pady=5, sticky="e")
        self.disconnect_button = ttk.Button(button_frame, text="断开连接", command=self.disconnect_mqtt, state=tk.DISABLED, style="Disconnect.TButton"); self.disconnect_button.grid(row=0, column=1, padx=20, pady=5, sticky="w")

        simulation_frame = ttk.Frame(root, style="Custom.TFrame")
        simulation_frame.pack(side=tk.BOTTOM, pady=(0,10), fill="x")
        simulation_frame.columnconfigure(0, weight=1); simulation_frame.columnconfigure(1, weight=1); simulation_frame.columnconfigure(2, weight=1)
        self.select_file_button = ttk.Button(simulation_frame, text="选择CSV文件", command=self.select_csv_file, style="Simulate.TButton"); self.select_file_button.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.start_simulation_button = ttk.Button(simulation_frame, text="开始模拟", command=self.start_simulation, state=tk.DISABLED, style="Simulate.TButton"); self.start_simulation_button.grid(row=0, column=1, padx=10, pady=5)
        self.stop_simulation_button = ttk.Button(simulation_frame, text="停止模拟", command=self.stop_simulation_action, state=tk.DISABLED, style="Disconnect.TButton"); self.stop_simulation_button.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        self.csv_file_path = None
        self.simulation_delay = 0.2
        self.current_road_speed_limit = None

        self.current_data_text = tk.Text(root, height=10, bg="#1E2B37", fg=COLOR_TEXT, font=FONT_TEXT_AREA, relief="sunken", borderwidth=1, padx=10, pady=10)
        self.current_data_text.pack(pady=10, padx=20, fill="both", expand=True, side=tk.TOP)
        self.current_data_text.insert(tk.END, "等待数据或选择CSV文件开始模拟...\n"); self.current_data_text.config(state=tk.DISABLED)

        if not load_model_and_scaler():
            self.update_status("错误: 模型或Scaler加载失败", is_error=True, color=COLOR_ERROR)
            self.connect_button.config(state=tk.DISABLED)
            self.select_file_button.config(state=tk.DISABLED)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def select_csv_file(self):
        filepath = filedialog.askopenfilename(title="选择GPS特征CSV文件", filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")))
        if filepath:
            self.csv_file_path = filepath
            self.update_status(f"已选择文件: {filepath.split('/')[-1]}", color=COLOR_INFO)
            self.start_simulation_button.config(state=tk.NORMAL)
        else:
            self.csv_file_path = None
            self.start_simulation_button.config(state=tk.DISABLED)

    def _simulation_worker(self):
        global is_simulating
        is_simulating = True
        stop_simulation_event.clear()
        self.update_status("模拟进行中...", color=COLOR_INFO)
        self.start_simulation_button.config(state=tk.DISABLED); self.select_file_button.config(state=tk.DISABLED); self.stop_simulation_button.config(state=tk.NORMAL)
        self.disable_mqtt_controls()
        try:
            df = pd.read_csv(self.csv_file_path)
            if '时间戳' not in df.columns: df['时间戳'] = pd.to_datetime(time.time(), unit='s') + pd.to_timedelta(df.index * self.simulation_delay, unit='s')
            # *** 修改点 1: 检查 Lng 和 Lat 列 ***
            if not ('Lng' in df.columns and 'Lat' in df.columns):
                 messagebox.showwarning("模拟警告", "CSV文件中缺少 'Lng' 或 'Lat' 列，无法进行API限速判断。")
            for index, row in df.iterrows():
                if stop_simulation_event.is_set(): self.update_status("模拟已停止。", color=COLOR_WARNING); break
                data_dict = row.to_dict()
                self.root.after(0, self.process_data_point, data_dict)
                time.sleep(self.simulation_delay)
            else:
                if not stop_simulation_event.is_set(): self.update_status("模拟完成。", color=COLOR_SUCCESS)
        except FileNotFoundError: self.update_status(f"错误: CSV文件未找到: {self.csv_file_path}", is_error=True, color=COLOR_ERROR); messagebox.showerror("模拟错误", f"CSV文件未找到: {self.csv_file_path}")
        except pd.errors.EmptyDataError: self.update_status("错误: CSV文件为空。", is_error=True, color=COLOR_ERROR); messagebox.showerror("模拟错误", "选择的CSV文件为空。")
        except Exception as e: self.update_status(f"模拟出错: {e}", is_error=True, color=COLOR_ERROR); messagebox.showerror("模拟错误", f"处理CSV文件时发生错误: {e}")
        finally:
            is_simulating = False
            self.start_simulation_button.config(state=tk.NORMAL if self.csv_file_path else tk.DISABLED); self.select_file_button.config(state=tk.NORMAL); self.stop_simulation_button.config(state=tk.DISABLED)
            self.enable_mqtt_controls()
            if not stop_simulation_event.is_set() and not self.status_label.cget("text").endswith("模拟完成。"):
                 if not self.status_label.cget("text").startswith("状态: 错误"): self.update_status("模拟结束或中断。", color=COLOR_WARNING)

    def start_simulation(self):
        global simulation_thread
        if not self.csv_file_path: messagebox.showwarning("提示", "请先选择一个CSV文件."); return
        if is_simulating: messagebox.showinfo("提示", "模拟已经在运行中."); return
        data_buffer.clear()
        self.prediction_label.config(text="预测行为: -", foreground=COLOR_TEXT); self.overspeed_label.config(text="")
        self.current_data_text.config(state=tk.NORMAL); self.current_data_text.delete(1.0, tk.END); self.current_data_text.insert(tk.END, "开始从CSV文件模拟数据...\n"); self.current_data_text.config(state=tk.DISABLED)
        simulation_thread = threading.Thread(target=self._simulation_worker, daemon=True); simulation_thread.start()

    def stop_simulation_action(self):
        if is_simulating and simulation_thread and simulation_thread.is_alive(): stop_simulation_event.set(); self.update_status("正在停止模拟...", color=COLOR_WARNING)
        else: self.update_status("模拟未在运行。", color=COLOR_INFO)

    def disable_mqtt_controls(self): self.broker_entry.config(state=tk.DISABLED); self.port_entry.config(state=tk.DISABLED); self.topic_entry.config(state=tk.DISABLED); self.connect_button.config(state=tk.DISABLED); self.disconnect_button.config(state=tk.DISABLED)
    def enable_mqtt_controls(self): self.broker_entry.config(state=tk.NORMAL); self.port_entry.config(state=tk.NORMAL); self.topic_entry.config(state=tk.NORMAL); self.connect_button.config(state=tk.NORMAL if model and scaler else tk.DISABLED)

    def update_status(self, message, is_error=False, color=None):
        fg_color = COLOR_TEXT
        if color: fg_color = color
        elif is_error: fg_color = COLOR_ERROR
        else:
            if "已连接" in message and "失败" not in message: fg_color = COLOR_SUCCESS
            elif "断开" in message or "意外" in message or "停止" in message or "中断" in message: fg_color = COLOR_WARNING
            elif "模拟完成" in message: fg_color = COLOR_SUCCESS
            elif "模拟进行中" in message or "已选择文件" in message: fg_color = COLOR_INFO
        self.status_label.config(text=f"状态: {message}", foreground=fg_color); print(f"UI状态: {message}")

    def update_prediction_display(self, current_point_dict, prediction_idx, overspeed_info=None):
        self.current_data_text.config(state=tk.NORMAL); self.current_data_text.delete(1.0, tk.END)
        self.current_data_text.insert(tk.END, "最新接收数据点 (用于形成当前窗口):\n\n", "header_tag"); self.current_data_text.tag_configure("header_tag", font=FONT_BOLD, foreground=COLOR_ACCENT)
        if current_point_dict:
            # *** 修改点 3: 确保显示的是转换后的 longitude 和 latitude (如果需要显示原始 Lng/Lat, 则需调整) ***
            # 为了统一，我们假设 process_data_point 内部会处理 Lng/Lat 到 longitude/latitude 的映射（如果需要）
            # 或者直接在 current_point_dict 中使用 Lng/Lat。这里我们假设 current_point_dict 仍包含 Lng/Lat。
            display_keys = ["gps速度", "加速度", "曲折度", "vss速度", "ACC状态", "与正北方向夹角", "时间戳", "Lng", "Lat"]
            for key in display_keys:
                if key in current_point_dict:
                    value = current_point_dict[key]
                    if key == "时间戳" and isinstance(value, (pd.Timestamp, np.datetime64)):
                         try: value_str = pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                         except: value_str = str(value)
                    elif isinstance(value, float): value_str = f"{value:.2f}" if key not in ["Lng", "Lat"] else f"{value:.6f}"
                    else: value_str = str(value)
                    self.current_data_text.insert(tk.END, f"{key + ':':<20}", "key_tag")
                    self.current_data_text.insert(tk.END, f"{value_str}\n", "value_tag")
            self.current_data_text.tag_configure("key_tag", foreground=COLOR_SECONDARY, font=FONT_TEXT_AREA)
            self.current_data_text.tag_configure("value_tag", foreground=COLOR_TEXT, font=FONT_TEXT_AREA)
            self.current_data_text.insert(tk.END, "\n-------\n", "separator_tag")
            self.current_data_text.insert(tk.END, f"当前数据缓冲区大小: {len(data_buffer)}/{WINDOW_SIZE}\n", "buffer_tag")
            self.current_data_text.tag_configure("separator_tag", foreground=COLOR_SECONDARY); self.current_data_text.tag_configure("buffer_tag", foreground=COLOR_ACCENT, font=(FONT_TEXT_AREA[0], FONT_TEXT_AREA[1], "italic"))
        self.current_data_text.config(state=tk.DISABLED)

        if prediction_idx != -1:
            behavior = CLASS_LABELS.get(prediction_idx, "未知行为")
            self.prediction_label.config(text=f"预测行为: {behavior}")
            if prediction_idx in ABNORMAL_CLASSES and behavior != CLASS_LABELS[1]: self.prediction_label.config(foreground=COLOR_ERROR)
            elif behavior == CLASS_LABELS[1]: self.prediction_label.config(foreground=COLOR_WARNING)
            else: self.prediction_label.config(foreground=COLOR_SUCCESS)
        else: self.prediction_label.config(text="预测行为: (等待数据或窗口未满)", foreground=COLOR_TEXT)

        if overspeed_info:
            self.overspeed_label.config(text=overspeed_info["message"])
            if overspeed_info["is_overspeeding"]:
                self.overspeed_label.config(foreground=COLOR_OVERSPEED)
                if self.prediction_label.cget("foreground") != str(COLOR_ERROR): self.prediction_label.config(text=f"预测行为: {CLASS_LABELS.get(prediction_idx, '未知')} (API超速!)", foreground=COLOR_OVERSPEED)
            else: self.overspeed_label.config(foreground=COLOR_INFO)
        else: self.overspeed_label.config(text="")

    def process_data_point(self, raw_data_dict):
        global data_buffer
        if not isinstance(raw_data_dict, dict): self.update_status(f"错误: 数据格式无效", is_error=True, color=COLOR_ERROR); return

        overspeed_check_result = None
        current_gps_speed = raw_data_dict.get("gps速度")
        # *** 修改点 2: 从 raw_data_dict 中提取 Lng 和 Lat ***
        latitude_val = raw_data_dict.get("Lat") # 使用 "Lat"
        longitude_val = raw_data_dict.get("Lng") # 使用 "Lng"

        try:
            current_gps_speed_float = float(current_gps_speed if current_gps_speed is not None else 0)
            lat_float = float(latitude_val if latitude_val is not None else 0)
            lon_float = float(longitude_val if longitude_val is not None else 0)
        except (ValueError, TypeError):
            current_gps_speed_float = 0; lat_float = None; lon_float = None
            print("警告: GPS速度或经纬度数据无效，无法判断超速。")

        if lat_float is not None and lon_float is not None and current_gps_speed_float > 0:
            road_speed_limit = get_speed_limit_from_api(lat_float, lon_float)
            if road_speed_limit is not None:
                self.current_road_speed_limit = road_speed_limit
                if current_gps_speed_float > road_speed_limit * 1.1: overspeed_check_result = {"is_overspeeding": True, "message": f"API超速! 当前车速: {current_gps_speed_float:.1f} km/h, 限速: {road_speed_limit} km/h"}
                else: overspeed_check_result = {"is_overspeeding": False, "message": f"API限速: {road_speed_limit} km/h, 车速正常"}
            elif self.current_road_speed_limit is not None:
                 if current_gps_speed_float > self.current_road_speed_limit * 1.1: overspeed_check_result = {"is_overspeeding": True, "message": f"API超速! (缓存限速) 当前车速: {current_gps_speed_float:.1f} km/h, 限速: {self.current_road_speed_limit} km/h"}
                 else: overspeed_check_result = {"is_overspeeding": False, "message": f"API限速: {self.current_road_speed_limit} km/h (缓存), 车速正常"}

        model_input_feature_values = []
        acc_status_val = 0
        if "ACC状态" in raw_data_dict:
            acc_val = raw_data_dict["ACC状态"]
            if isinstance(acc_val, str) and acc_val.lower() == "acc开": acc_status_val = 1
            elif isinstance(acc_val, (int, float)) and acc_val == 1: acc_status_val = 1
        for col_name in FEATURE_COLS:
            if col_name == "ACC状态": model_input_feature_values.append(acc_status_val)
            else:
                try:
                    value_from_dict = raw_data_dict.get(col_name)
                    if pd.isna(value_from_dict): value = 0.0
                    else: value = float(value_from_dict)
                except (ValueError, TypeError): value = 0.0
                model_input_feature_values.append(value)
        data_buffer.append(model_input_feature_values)

        prediction_idx = -1
        if len(data_buffer) == WINDOW_SIZE:
            window_data_np = np.array(list(data_buffer))
            if scaler:
                try:
                    window_data_scaled = scaler.transform(window_data_np)
                    prediction_idx = predict_behavior(window_data_scaled)
                except Exception as e: print(f"标准化或预测时出错: {e}"); self.update_status(f"错误: 预测失败 - {e}", is_error=True, color=COLOR_ERROR)
            else: print("Scaler未加载，无法标准化"); self.update_status("错误: Scaler未加载", is_error=True, color=COLOR_ERROR)
        self.update_prediction_display(raw_data_dict, prediction_idx, overspeed_check_result)

    def on_connect(self, client, userdata, flags, rc, properties=None):
        global is_connected_to_mqtt
        if rc == 0:
            self.update_status(f"已连接到 Broker!", color=COLOR_SUCCESS); is_connected_to_mqtt = True; topic = self.topic_entry.get(); client.subscribe(topic)
            self.update_status(f"已订阅 Topic: {topic}", color=COLOR_SUCCESS); data_buffer.clear(); self.disconnect_button.config(state=tk.NORMAL); self.connect_button.config(state=tk.DISABLED)
        else: self.update_status(f"连接失败, 返回码: {rc}", is_error=True, color=COLOR_ERROR); is_connected_to_mqtt = False; self.disconnect_button.config(state=tk.DISABLED); self.connect_button.config(state=tk.NORMAL)

    def on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode('utf-8'); data_dict = json.loads(payload_str)
            self.root.after(0, self.process_data_point, data_dict)
        except json.JSONDecodeError: print(f"错误: 无法解析JSON: {msg.payload}"); self.update_status("错误: 接收到无效JSON数据", is_error=True, color=COLOR_ERROR)
        except Exception as e: print(f"处理消息时出错: {e}"); self.update_status(f"错误: {e}", is_error=True, color=COLOR_ERROR)

    def on_disconnect(self, client, userdata, rc, properties=None):
        global is_connected_to_mqtt; is_connected_to_mqtt = False
        if rc != 0: self.update_status(f"意外断开 (rc: {rc}). 请检查Broker.", is_error=True, color=COLOR_ERROR)
        else: self.update_status("已断开连接", color=COLOR_WARNING)
        self.connect_button.config(state=tk.NORMAL); self.disconnect_button.config(state=tk.DISABLED); self.prediction_label.config(text="预测行为: -", foreground=COLOR_TEXT); self.overspeed_label.config(text="")

    def connect_mqtt(self):
        global mqtt_client, is_connected_to_mqtt
        if is_simulating: messagebox.showinfo("提示", "模拟进行中，请先停止模拟再连接MQTT."); return
        if is_connected_to_mqtt: messagebox.showinfo("提示", "已经连接到MQTT Broker."); return
        broker_address = self.broker_entry.get(); broker_port_str = self.port_entry.get(); topic = self.topic_entry.get()
        if not broker_address or not broker_port_str or not topic: messagebox.showerror("输入错误", "请输入Broker地址、端口和Topic."); return
        try: broker_port = int(broker_port_str)
        except ValueError: messagebox.showerror("输入错误", "端口号必须是数字."); return
        try:
            import paho.mqtt.client as mqtt_module
            mqtt_client = mqtt_module.Client(mqtt_module.CallbackAPIVersion.VERSION2)
            mqtt_client.on_connect = self.on_connect; mqtt_client.on_message = self.on_message; mqtt_client.on_disconnect = self.on_disconnect
            self.update_status(f"正在连接到 {broker_address}:{broker_port}...", color=COLOR_TEXT)
            mqtt_client.connect_async(broker_address, broker_port, 60); mqtt_client.loop_start()
        except ImportError: messagebox.showerror("错误", "Paho MQTT库未找到。请先安装: pip install paho-mqtt"); self.update_status("错误: MQTT库加载失败", is_error=True, color=COLOR_ERROR)
        except Exception as e: self.update_status(f"连接MQTT失败: {e}", is_error=True, color=COLOR_ERROR); messagebox.showerror("MQTT 连接错误", f"连接失败: {e}"); self.connect_button.config(state=tk.NORMAL); self.disconnect_button.config(state=tk.DISABLED)

    def disconnect_mqtt(self):
        global mqtt_client, is_connected_to_mqtt
        if mqtt_client and (is_connected_to_mqtt or (hasattr(mqtt_client, '_thread') and mqtt_client._thread is not None)):
            self.update_status("正在断开连接...", color=COLOR_WARNING)
            if hasattr(mqtt_client, '_thread') and mqtt_client._thread is not None: mqtt_client.loop_stop()
            # Check connection state before disconnecting for paho-mqtt 1.x and 2.x
            is_really_connected = False
            if hasattr(mqtt_client, 'is_connected') and callable(mqtt_client.is_connected): # paho-mqtt 1.x
                is_really_connected = mqtt_client.is_connected()
            elif hasattr(mqtt_client, '_state'): # paho-mqtt 2.x (approximate check)
                try:
                    import paho.mqtt.client as mqtt_defs # For MQTT_CS_CONNECTED
                    is_really_connected = (mqtt_client._state == mqtt_defs.MQTT_CS_CONNECTED)
                except: pass # Ignore if paho.mqtt.client cannot be imported here

            if is_really_connected:
                mqtt_client.disconnect()
        else: self.update_status("未连接或已断开", color=COLOR_TEXT)
        is_connected_to_mqtt = False; self.connect_button.config(state=tk.NORMAL); self.disconnect_button.config(state=tk.DISABLED); self.overspeed_label.config(text="")

    def on_closing(self):
        if is_simulating: stop_simulation_event.set();
        if simulation_thread and simulation_thread.is_alive(): simulation_thread.join(timeout=1)
        self.disconnect_mqtt(); self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
