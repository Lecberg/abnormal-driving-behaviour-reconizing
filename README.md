# Abnormal Driving Behavior Prototype

This project is a prototype for abnormal driving behavior detection from GPS trajectory data.

It includes:

- A Python desktop client.
- A Python API for the browser dashboard.
- A React dashboard interface.
- Model files and example data for local testing.

## Requirements

Install these tools before running the project:

- Python 3.12, or another Python version that supports the required PyTorch package.
- Node.js and npm for the browser dashboard.
- Rust and Cargo only if you want to run the optional Tauri desktop wrapper.

## Setup

Clone the repository and enter the project folder:

```powershell
git clone https://github.com/Lecberg/abnormal-driving-behaviour-reconizing.git
cd abnormal-driving-behaviour-reconizing
```

Create a Python virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Python Desktop Client

Run this command from the project root:

```powershell
python abnormal_driving_client\client_app.py
```

## Browser Dashboard Demo

The browser dashboard is the recommended presentation view.

![Web dashboard screenshot](docs/web-dashboard-screenshot.png)

Start the Python API from the project root:

```powershell
python -m uvicorn abnormal_driving_client.web_server:app --host 127.0.0.1 --port 8000
```

Start the web dashboard in another terminal:

```powershell
cd abnormal_driving_client_tauri
npm install
npm run dev
```

Then open this address in your browser:

```text
http://127.0.0.1:5173
```

The dashboard loads `gps_sample.csv` by default. You can also upload another CSV file from the page.

## Optional Tauri Desktop Wrapper

The React dashboard can also run inside a Tauri desktop window.

Install the frontend dependencies first:

```powershell
cd abnormal_driving_client_tauri
npm install
```

Then run:

```powershell
npm run tauri dev
```

If Tauri cannot find Python, set the `PYTHON` environment variable to your Python executable.

```powershell
$env:PYTHON="path\to\python.exe"
npm run tauri dev
```

## Optional Map API

The client can use Amap reverse geocoding to help check road speed limits.

Set the API key before running the client:

```powershell
$env:AMAP_API_KEY="your_api_key_here"
python abnormal_driving_client\client_app.py
```

If `AMAP_API_KEY` is not set, the client still runs. It only disables API-based speed-limit checking.

## Main Files

- `abnormal_driving_client/client_app.py`: main Tkinter client.
- `abnormal_driving_client/backend_runtime.py`: reusable Python runtime used by the React + Tauri client.
- `abnormal_driving_client/backend_service.py`: JSON bridge between Tauri and the Python runtime.
- `abnormal_driving_client_tauri/`: React + Tauri desktop client.
- `abnormal_driving_client/model_definition.py`: runtime model definition and feature list.
- `abnormal_driving_client/best_model.pth`: trained model weights.
- `abnormal_driving_client/scaler.gz`: saved feature scaler.
- `abnormal_driving_client/mqtt_publisher_dummy.py`: test MQTT publisher.

Older files such as `client_app2.py` and `3.py` are historical versions. Use `client_app.py` as the official client.
