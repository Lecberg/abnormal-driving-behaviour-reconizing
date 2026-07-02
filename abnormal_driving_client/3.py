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
import logging
from queue import Queue

# 从你的模型文件中导入模型类和相关常量
from CNNLSTMModel import CNNLSTMModel, FEATURE_COLS, WINDOW_SIZE, DEVICE

# --- 全局常量 ---
MODEL_PATH = "abnormal_driving_client/best_model.pth"
SCALER_PATH = "abnormal_driving_client/scaler.gz"
NUM_FEATURES = len(FEATURE_COLS)
NUM_CLASSES = 4

# 类别标签和优先级
CLASS_LABELS = {0: "正常驾驶", 1: "高速行驶", 2: "急加速/急减速", 3: "曲折行驶"}
BEHAVIOR_PRIORITY = {"曲折行驶": 0, "急加速/急减速": 1, "高速行驶": 2, "API超速": 3}
MODEL_INDEX_TO_PRIORITY_KEY = {0: "正常驾驶", 1: "高速行驶", 2: "急加速/急减速", 3: "曲折行驶"}

# 地图API配置
MAP_API_PROVIDER = "amap"
AMAP_API_KEY = "YOUR_AMAP_WEB_SERVICE_API_KEY"  # <--- 在这里替换成你的高德Web服务API Key
AMAP_REVERSE_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"

# UI颜色和字体
COLOR_PRIMARY = "#2E3F4F"
COLOR_SECONDARY = "#4A6F8A"
COLOR_ACCENT = "#F7B733"
COLOR_TEXT = "#FFFFFF"
COLOR_SUCCESS = "#2ECC71"
COLOR_ERROR = "#E74C3C"
COLOR_WARNING = "#F39C12"
COLOR_OVERSPEED_API = "#FF00FF"
COLOR_INFO = "#3498DB"

FONT_FAMILY_CN = "微软雅黑"
FONT_NORMAL = (FONT_FAMILY_CN, 10)
FONT_BOLD = (FONT_FAMILY_CN, 12, "bold")
FONT_LARGE_BOLD = (FONT_FAMILY_CN, 16, "bold")
FONT_STATUS = (FONT_FAMILY_CN, 11)
FONT_PREDICTION = (FONT_FAMILY_CN, 18, "bold")
FONT_TEXT_AREA = ("Consolas", 10)
FONT_OVERSPEED_INFO_UI = (FONT_FAMILY_CN, 10, "italic")

# --- 全局变量 ---
model = None
scaler = None
data_buffer = deque(maxlen=WINDOW_SIZE)
simulation_thread = None
stop_simulation_event = threading.Event()
is_simulating = False
api_result_queue = Queue()
last_api_call_time = 0
API_CALL_INTERVAL = 5

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scaler():
    global model, scaler
    try:
        model = CNNLSTMModel(input_size=NUM_FEATURES, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logging.info("模型加载成功!")
        scaler = joblib.load(SCALER_PATH)
        logging.info("Scaler加载成功!")
        return True
    except FileNotFoundError:
        logging.error(f"模型文件 '{MODEL_PATH}' 或 Scaler文件 '{SCALER_PATH}' 未找到。")
        messagebox.showerror("加载错误", f"模型或Scaler文件未找到。\n请检查路径:\n{MODEL_PATH}\n{SCALER_PATH}")
        return False
    except Exception as e:
        logging.error(f"加载模型或Scaler失败: {e}")
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
        logging.error(f"预测出错: {e}")
        return -1

def get_speed_limit_from_api(latitude, longitude):
    global last_api_call_time
    current_time = time.time()
    if current_time - last_api_call_time < API_CALL_INTERVAL:
        return None
    last_api_call_time = current_time
    logging.info(f"正在为坐标 ({latitude}, {longitude}) 调用地图API...")
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
                    if speed_limit_str and speed_limit_str.isdigit():
                        speed_limit = int(speed_limit_str)
                        logging.info(f"API成功获取限速: {speed_limit} km/h")
                        return speed_limit
            logging.warning(f"API未返回有效的限速信息。响应: {data}")
            return None
        except requests.exceptions.RequestException as e: logging.error(f"高德地图API请求失败: {e}"); return None
        except (json.JSONDecodeError, KeyError, IndexError) as e: logging.error(f"解析高德地图API响应失败: {e}"); return None
    else: logging.warning(f"未知的地图API提供商: {MAP_API_PROVIDER}"); return None

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("异常驾驶行为检测客户端 (模拟模式)")
        self.root.geometry("750x650") # 调整窗口大小
        self.root.configure(bg=COLOR_PRIMARY)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", background=COLOR_PRIMARY, foreground=COLOR_TEXT, font=FONT_NORMAL, padding=5)
        self.style.configure("Title.TLabel", font=FONT_LARGE_BOLD, foreground=COLOR_ACCENT)
        self.style.configure("Status.TLabel", font=FONT_STATUS, padding=10)
        self.style.configure("Prediction.TLabel", font=FONT_PREDICTION, padding=10)
        self.style.configure("Overspeed.TLabel", font=FONT_OVERSPEED_INFO_UI, padding=5)
        self.style.configure("TButton", font=FONT_BOLD, padding=(10, 5))
        self.style.map("TButton", foreground=[('active', COLOR_PRIMARY), ('!disabled', COLOR_TEXT)], background=[('active', COLOR_ACCENT), ('!disabled', COLOR_SECONDARY)], relief=[('pressed', 'sunken'), ('!pressed', 'raised')])
        self.style.configure("Disconnect.TButton", background=COLOR_WARNING)
        self.style.map("Disconnect.TButton", background=[('active', COLOR_ERROR)])
        self.style.configure("Simulate.TButton", background=COLOR_INFO)
        self.style.map("Simulate.TButton", background=[('active', COLOR_ACCENT)])
        self.style.configure("TLabelframe", background=COLOR_PRIMARY, relief="groove", borderwidth=2)
        self.style.configure("TLabelframe.Label", background=COLOR_PRIMARY, foreground=COLOR_ACCENT, font=FONT_BOLD)
        self.style.configure("Custom.TFrame", background=COLOR_PRIMARY)
        self.style.configure("TScale", background=COLOR_PRIMARY)

        title_label = ttk.Label(root, text="异常驾驶行为实时监测系统", style="Title.TLabel")
        title_label.pack(pady=(20,10), side=tk.TOP)

        status_prediction_frame = ttk.Frame(root, style="Custom.TFrame", padding=10)
        status_prediction_frame.pack(pady=5, fill="x", padx=20, side=tk.TOP)
        self.status_label = ttk.Label(status_prediction_frame, text="状态: 等待操作", style="Status.TLabel", anchor="center"); self.status_label.pack(pady=(0,5), fill="x")
        self.prediction_label = ttk.Label(status_prediction_frame, text="预测行为: -", style="Prediction.TLabel", anchor="center"); self.prediction_label.pack(pady=5, fill="x")
        self.overspeed_label = ttk.Label(status_prediction_frame, text="", style="Overspeed.TLabel", anchor="center"); self.overspeed_label.pack(pady=(0,5), fill="x")

        simulation_frame = ttk.LabelFrame(root, text="模拟控制", style="TLabelframe")
        simulation_frame.pack(side=tk.BOTTOM, pady=20, padx=20, fill="x")
        simulation_frame.columnconfigure(1, weight=1)
        self.select_file_button = ttk.Button(simulation_frame, text="选择CSV文件", command=self.select_csv_file, style="Simulate.TButton"); self.select_file_button.grid(row=0, column=0, padx=10, pady=5)
        self.start_simulation_button = ttk.Button(simulation_frame, text="开始模拟", command=self.start_simulation, state=tk.DISABLED, style="Simulate.TButton"); self.start_simulation_button.grid(row=0, column=1, padx=10, pady=5)
        self.stop_simulation_button = ttk.Button(simulation_frame, text="停止模拟", command=self.stop_simulation_action, state=tk.DISABLED, style="Disconnect.TButton"); self.stop_simulation_button.grid(row=0, column=2, padx=10, pady=5)
        ttk.Label(simulation_frame, text="模拟速度:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.simulation_speed_scale = ttk.Scale(simulation_frame, from_=0.05, to=2.0, orient=tk.HORIZONTAL, command=self.update_simulation_speed)
        self.simulation_speed_scale.set(0.2); self.simulation_speed_scale.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        self.simulation_delay = 0.2

        self.csv_file_path = None
        self.current_road_speed_limit = None

        self.current_data_text = tk.Text(root, height=10, bg="#1E2B37", fg=COLOR_TEXT, font=FONT_TEXT_AREA, relief="sunken", borderwidth=1, padx=10, pady=10)
        self.current_data_text.pack(pady=10, padx=20, fill="both", expand=True, side=tk.TOP)
        self.current_data_text.insert(tk.END, "请选择CSV文件并开始模拟...\n"); self.current_data_text.config(state=tk.DISABLED)

        if not load_model_and_scaler():
            self.update_status("错误: 模型或Scaler加载失败", is_error=True)
            self.select_file_button.config(state=tk.DISABLED)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_api_queue()

    def update_simulation_speed(self, value):
        self.simulation_delay = float(value)

    def check_api_queue(self):
        try:
            while not api_result_queue.empty():
                result = api_result_queue.get_nowait()
                self.current_road_speed_limit = result["limit"]
                logging.info(f"从队列接收到API结果: 限速 {self.current_road_speed_limit}")
        except Exception as e:
            logging.error(f"检查API队列时出错: {e}")
        finally:
            self.root.after(100, self.check_api_queue)

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
        try:
            df = pd.read_csv(self.csv_file_path)
            if '时间戳' not in df.columns: df['时间戳'] = pd.to_datetime(time.time(), unit='s') + pd.to_timedelta(df.index * self.simulation_delay, unit='s')
            if not ('Lng' in df.columns and 'Lat' in df.columns):
                 messagebox.showwarning("模拟警告", "CSV文件中缺少 'Lng' 或 'Lat' 列，无法进行API限速判断。")
            for index, row in df.iterrows():
                if stop_simulation_event.is_set(): self.update_status("模拟已停止。", color=COLOR_WARNING); break
                data_dict = row.to_dict()
                self.root.after(0, self.process_data_point, data_dict)
                time.sleep(self.simulation_delay)
            else:
                if not stop_simulation_event.is_set(): self.update_status("模拟完成。", color=COLOR_SUCCESS)
        except FileNotFoundError: self.update_status(f"错误: CSV文件未找到", is_error=True); messagebox.showerror("模拟错误", f"CSV文件未找到: {self.csv_file_path}")
        except pd.errors.EmptyDataError: self.update_status("错误: CSV文件为空", is_error=True); messagebox.showerror("模拟错误", "选择的CSV文件为空。")
        except Exception as e: self.update_status(f"模拟出错: {e}", is_error=True); messagebox.showerror("模拟错误", f"处理CSV文件时发生错误: {e}")
        finally:
            is_simulating = False
            self.start_simulation_button.config(state=tk.NORMAL if self.csv_file_path else tk.DISABLED); self.select_file_button.config(state=tk.NORMAL); self.stop_simulation_button.config(state=tk.DISABLED)
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

    def update_status(self, message, is_error=False, color=None):
        fg_color = COLOR_TEXT
        if color: fg_color = color
        elif is_error: fg_color = COLOR_ERROR
        else:
            if "已连接" in message and "失败" not in message: fg_color = COLOR_SUCCESS
            elif "断开" in message or "意外" in message or "停止" in message or "中断" in message: fg_color = COLOR_WARNING
            elif "模拟完成" in message: fg_color = COLOR_SUCCESS
            elif "模拟进行中" in message or "已选择文件" in message: fg_color = COLOR_INFO
        self.status_label.config(text=f"状态: {message}", foreground=fg_color); logging.info(f"UI状态: {message}")

    def update_prediction_display(self, current_point_dict, prediction_idx, overspeed_info=None, road_limit=None):
        self.current_data_text.config(state=tk.NORMAL); self.current_data_text.delete(1.0, tk.END)
        self.current_data_text.insert(tk.END, "最新接收数据点 (用于形成当前窗口):\n\n", "header_tag"); self.current_data_text.tag_configure("header_tag", font=FONT_BOLD, foreground=COLOR_ACCENT)
        if current_point_dict:
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

        final_behavior_text = "预测行为: (等待数据或窗口未满)"
        final_behavior_color = COLOR_TEXT
        current_priority = float('inf')
        model_predicted_behavior_key = None
        if prediction_idx != -1:
            model_predicted_behavior_key = MODEL_INDEX_TO_PRIORITY_KEY.get(prediction_idx)
            if model_predicted_behavior_key and model_predicted_behavior_key != "正常驾驶":
                priority = BEHAVIOR_PRIORITY.get(model_predicted_behavior_key, float('inf'))
                if priority < current_priority:
                    current_priority = priority
                    final_behavior_text = f"预测行为: {model_predicted_behavior_key}"
                    if model_predicted_behavior_key == "高速行驶": final_behavior_color = COLOR_WARNING
                    else: final_behavior_color = COLOR_ERROR
            elif model_predicted_behavior_key == "正常驾驶":
                final_behavior_text = "预测行为: 正常驾驶"
                final_behavior_color = COLOR_SUCCESS

        api_is_overspeeding = overspeed_info and overspeed_info["is_overspeeding"]
        if api_is_overspeeding:
            api_overspeed_priority = BEHAVIOR_PRIORITY.get("API超速", float('inf'))
            if api_overspeed_priority < current_priority or \
               (model_predicted_behavior_key == "正常驾驶") or \
               (model_predicted_behavior_key == "高速行驶" and api_overspeed_priority <= BEHAVIOR_PRIORITY.get("高速行驶")):
                current_priority = api_overspeed_priority
                final_behavior_text = f"预测行为: API超速"
                final_behavior_color = COLOR_OVERSPEED_API
        self.prediction_label.config(text=final_behavior_text, foreground=final_behavior_color)

        overspeed_label_text = ""
        overspeed_label_color = COLOR_TEXT
        if road_limit is not None:
            overspeed_label_text = f"道路限速: {road_limit} km/h"
            overspeed_label_color = COLOR_INFO
            if overspeed_info and overspeed_info["is_overspeeding"]:
                overspeed_label_text += f" (当前: {overspeed_info.get('speed', 'N/A')} km/h - 超速!)"
                overspeed_label_color = COLOR_OVERSPEED_API
            elif overspeed_info and not overspeed_info["is_overspeeding"]:
                 overspeed_label_text += " (车速正常)"
        elif overspeed_info and overspeed_info["is_overspeeding"]:
            overspeed_label_text = f"API超速! (当前: {overspeed_info.get('speed', 'N/A')} km/h)"
            overspeed_label_color = COLOR_OVERSPEED_API
        self.overspeed_label.config(text=overspeed_label_text, foreground=overspeed_label_color)

    def process_data_point(self, raw_data_dict):
        global data_buffer
        if not isinstance(raw_data_dict, dict): self.update_status(f"错误: 数据格式无效", is_error=True); return

        overspeed_check_result = None
        road_speed_limit_for_ui = None
        current_gps_speed = raw_data_dict.get("gps速度")
        latitude_val = raw_data_dict.get("Lat")
        longitude_val = raw_data_dict.get("Lng")

        try:
            current_gps_speed_float = float(current_gps_speed if current_gps_speed is not None else 0)
            lat_float = float(latitude_val if latitude_val is not None else 0)
            lon_float = float(longitude_val if longitude_val is not None else 0)
        except (ValueError, TypeError):
            current_gps_speed_float = 0; lat_float = None; lon_float = None
            logging.warning("GPS速度或经纬度数据无效，无法判断超速。")

        if lat_float is not None and lon_float is not None and current_gps_speed_float >= 0:
            api_thread = threading.Thread(target=self.get_speed_limit_async, args=(lat_float, lon_float), daemon=True)
            api_thread.start()
            if self.current_road_speed_limit is not None:
                road_speed_limit_for_ui = self.current_road_speed_limit
                if current_gps_speed_float > self.current_road_speed_limit * 1.1:
                    overspeed_check_result = {"is_overspeeding": True, "speed": f"{current_gps_speed_float:.1f}"}
                else:
                    overspeed_check_result = {"is_overspeeding": False, "speed": f"{current_gps_speed_float:.1f}"}

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
                except Exception as e: logging.error(f"标准化或预测时出错: {e}"); self.update_status(f"错误: 预测失败 - {e}", is_error=True)
            else: logging.warning("Scaler未加载，无法标准化"); self.update_status("错误: Scaler未加载", is_error=True)
        self.update_prediction_display(raw_data_dict, prediction_idx, overspeed_check_result, self.current_road_speed_limit)

    def get_speed_limit_async(self, latitude, longitude):
        speed_limit = get_speed_limit_from_api(latitude, longitude, API_CALL_INTERVAL)
        if speed_limit is not None:
            api_result_queue.put({"lat": latitude, "lon": longitude, "limit": speed_limit})

    def on_closing(self):
        if is_simulating:
            stop_simulation_event.set()
            if simulation_thread and simulation_thread.is_alive():
                simulation_thread.join(timeout=1)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

