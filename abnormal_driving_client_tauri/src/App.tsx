import {
  Activity,
  AlertTriangle,
  BarChart3,
  Car,
  CheckCircle2,
  FileSpreadsheet,
  Gauge,
  Languages,
  Pause,
  Play,
  RotateCcw,
  Upload,
} from "lucide-react";
import { ChangeEvent, CSSProperties, ReactNode, useEffect, useMemo, useRef, useState } from "react";

type Language = "en" | "zh";
type StatusKind = "success" | "warning" | "danger" | "error" | "info" | "muted" | "loading";

type PredictionState = {
  label: string;
  message: string;
  confidence: number | null;
  classIndex: number | null;
  kind: StatusKind;
  warningSource: string;
  apiOverspeed: boolean;
};

type RuntimeState = {
  language: Language;
  status: { message: string; kind: StatusKind };
  mqtt: { connected: boolean };
  csv: { selectedPath: string | null; selectedName: string; simulating: boolean };
  window: { current: number; total: number };
  prediction: PredictionState;
  speedLimit: { limit: number | null; gpsSpeed: number | null; message: string; kind: StatusKind };
  latestData: Record<string, unknown>;
  modelApi: {
    modelLoaded: boolean;
    modelFile: string;
    apiEnabled: boolean;
    apiText: string;
    apiUrl: string;
  };
};

type BackendEvent = {
  event: string;
  payload?: Record<string, unknown>;
};

type StateResponse = {
  state: RuntimeState;
  warnings: Array<Record<string, unknown>>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const emptyPrediction: PredictionState = {
  label: "-",
  message: "-",
  confidence: null,
  classIndex: null,
  kind: "muted",
  warningSource: "",
  apiOverspeed: false,
};

const initialState: RuntimeState = {
  language: "en",
  status: { message: "Connecting to dashboard API...", kind: "loading" },
  mqtt: { connected: false },
  csv: { selectedPath: null, selectedName: "", simulating: false },
  window: { current: 0, total: 5 },
  prediction: emptyPrediction,
  speedLimit: { limit: null, gpsSpeed: null, message: "-", kind: "muted" },
  latestData: {},
  modelApi: {
    modelLoaded: false,
    modelFile: "best_model.pth",
    apiEnabled: false,
    apiText: "-",
    apiUrl: "",
  },
};

const labels = {
  en: {
    title: "Abnormal Driving Dashboard",
    subtitle: "CSV replay demo for driving behavior detection",
    model: "Model",
    csvReplay: "CSV Replay",
    language: "English",
    selectedCsv: "Selected CSV",
    noCsv: "Default sample or uploaded CSV",
    upload: "Upload CSV",
    start: "Start",
    continue: "Continue",
    pause: "Pause",
    reset: "Reset",
    prediction: "Live Prediction",
    confidence: "Confidence",
    warningSource: "Warning source",
    normal: "Normal",
    gpsSpeed: "GPS Speed",
    speedLimit: "Road Limit",
    replayProgress: "Replay Progress",
    warnings: "Warning Events",
    speedTrend: "Speed Trend",
    latestData: "Latest Data",
    noWarnings: "No warnings yet.",
    noData: "No data yet.",
    connected: "Connected",
    disconnected: "Disconnected",
    running: "Running",
    ready: "Ready",
  },
  zh: {
    title: "Abnormal Driving Dashboard",
    subtitle: "CSV replay demo for driving behavior detection",
    model: "Model",
    csvReplay: "CSV Replay",
    language: "English",
    selectedCsv: "Selected CSV",
    noCsv: "Default sample or uploaded CSV",
    upload: "Upload CSV",
    start: "Start",
    continue: "Continue",
    pause: "Pause",
    reset: "Reset",
    prediction: "Live Prediction",
    confidence: "Confidence",
    warningSource: "Warning source",
    normal: "Normal",
    gpsSpeed: "GPS Speed",
    speedLimit: "Road Limit",
    replayProgress: "Replay Progress",
    warnings: "Warning Events",
    speedTrend: "Speed Trend",
    latestData: "Latest Data",
    noWarnings: "No warnings yet.",
    noData: "No data yet.",
    connected: "Connected",
    disconnected: "Disconnected",
    running: "Running",
    ready: "Ready",
  },
} satisfies Record<Language, Record<string, string>>;

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function textValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function kindValue(value: unknown): StatusKind {
  const text = String(value);
  return ["success", "warning", "danger", "error", "info", "muted", "loading"].includes(text)
    ? (text as StatusKind)
    : "muted";
}

function classForKind(kind: StatusKind): string {
  return `is-${kind === "error" ? "danger" : kind}`;
}

function parsePrediction(payload: Record<string, unknown>): PredictionState {
  return {
    label: textValue(payload.label, "-"),
    message: textValue(payload.message, "-"),
    confidence: numberValue(payload.confidence),
    classIndex: numberValue(payload.classIndex),
    kind: kindValue(payload.kind),
    warningSource: textValue(payload.warningSource),
    apiOverspeed: Boolean(payload.apiOverspeed),
  };
}

function parseRuntime(payload: Record<string, unknown>): RuntimeState {
  const status = asRecord(payload.status);
  const csv = asRecord(payload.csv);
  const mqtt = asRecord(payload.mqtt);
  const windowState = asRecord(payload.window);
  const speedLimit = asRecord(payload.speedLimit);
  const modelApi = asRecord(payload.modelApi);

  return {
    language: payload.language === "zh" ? "zh" : "en",
    status: {
      message: textValue(status.message, initialState.status.message),
      kind: kindValue(status.kind),
    },
    mqtt: { connected: Boolean(mqtt.connected) },
    csv: {
      selectedPath: textValue(csv.selectedPath) || null,
      selectedName: textValue(csv.selectedName),
      simulating: Boolean(csv.simulating),
    },
    window: {
      current: numberValue(windowState.current) ?? 0,
      total: numberValue(windowState.total) ?? 5,
    },
    prediction: parsePrediction(asRecord(payload.prediction)),
    speedLimit: {
      limit: numberValue(speedLimit.limit),
      gpsSpeed: numberValue(speedLimit.gpsSpeed),
      message: textValue(speedLimit.message, "-"),
      kind: kindValue(speedLimit.kind),
    },
    latestData: asRecord(payload.latestData),
    modelApi: {
      modelLoaded: Boolean(modelApi.modelLoaded),
      modelFile: textValue(modelApi.modelFile, "best_model.pth"),
      apiEnabled: Boolean(modelApi.apiEnabled),
      apiText: textValue(modelApi.apiText, "-"),
      apiUrl: textValue(modelApi.apiUrl),
    },
  };
}

function displayNumber(value: number | null, suffix = ""): string {
  if (value === null) {
    return "-";
  }
  return `${value.toFixed(value % 1 === 0 ? 0 : 1)}${suffix}`;
}

function getGpsSpeed(row: Record<string, unknown>): number | null {
  for (const [key, value] of Object.entries(row)) {
    const normalized = key.toLowerCase();
    if (normalized.includes("gps") && normalized.includes("speed")) {
      const parsed = numberValue(value);
      if (parsed !== null) {
        return parsed;
      }
    }
  }
  return null;
}

const latestDataLabelMap: Record<string, string> = {
  vid_md5: "Vehicle ID",
  Lng: "Longitude",
  Lat: "Latitude",
  "\u8131\u654f\u8f66\u724c\u53f7": "Masked plate number",
  "\u8f66\u724c\u989c\u8272": "Plate color",
  "sim\u5361\u53f7": "SIM card number",
  "\u6807\u51c6\u65f6\u95f4": "Standard time",
  "\u7cfb\u7edf\u65f6\u95f4": "System time",
  "\u7701\u540d\u79f0": "Province",
  "\u5e02\u540d\u79f0": "City",
  "\u6d77\u62d4": "Altitude",
  "vss\u901f\u5ea6": "VSS speed",
  "gps\u901f\u5ea6": "GPS speed",
};

const latestDataValueMap: Record<string, string> = {
  "\u9ec4\u8272": "Yellow",
  "\u9655\u897f\u7701": "Shaanxi Province",
  "\u897f\u5b89\u5e02": "Xi'an",
};

const platePrefixMap: Record<string, string> = {
  "\u4eac": "Jing ",
  "\u6d25": "Jin ",
  "\u6caa": "Hu ",
  "\u6e1d": "Yu ",
  "\u5180": "Ji ",
  "\u8c6b": "Yu ",
  "\u4e91": "Yun ",
  "\u8fbd": "Liao ",
  "\u9ed1": "Hei ",
  "\u6e58": "Xiang ",
  "\u7696": "Wan ",
  "\u9c81": "Lu ",
  "\u65b0": "Xin ",
  "\u82cf": "Su ",
  "\u6d59": "Zhe ",
  "\u8d63": "Gan ",
  "\u9102": "E ",
  "\u6842": "Gui ",
  "\u7518": "Gan ",
  "\u664b": "Jin ",
  "\u8499": "Meng ",
  "\u9655": "Shaan ",
  "\u5409": "Ji ",
  "\u95fd": "Min ",
  "\u8d35": "Gui ",
  "\u7ca4": "Yue ",
  "\u9752": "Qing ",
  "\u85cf": "Zang ",
  "\u5ddd": "Chuan ",
  "\u5b81": "Ning ",
  "\u743c": "Qiong ",
};

function latestDataLabel(key: string): string {
  return latestDataLabelMap[key] ?? key;
}

function latestDataValue(value: unknown): string {
  const text = String(value ?? "-");
  const mapped = latestDataValueMap[text] ?? text;
  return mapped.replace(/[\u4e00-\u9fff]/g, (character) => platePrefixMap[character] ?? "");
}

function predictionTitleSize(text: string): string {
  const length = text.trim().length;
  if (length <= 18) {
    return "34px";
  }
  if (length <= 24) {
    return "28px";
  }
  if (length <= 32) {
    return "20px";
  }
  return "18px";
}

export default function App() {
  const [runtime, setRuntime] = useState<RuntimeState>(initialState);
  const [warnings, setWarnings] = useState<Array<Record<string, unknown>>>([]);
  const [speedPoints, setSpeedPoints] = useState<number[]>([]);
  const [apiOnline, setApiOnline] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [replayPaused, setReplayPaused] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const language = runtime.language;
  const t = labels[language];
  const progress = runtime.window.total ? Math.round((runtime.window.current / runtime.window.total) * 100) : 0;
  const gpsSpeed = runtime.speedLimit.gpsSpeed ?? getGpsSpeed(runtime.latestData);
  const confidence = runtime.prediction.confidence === null ? "-" : `${Math.round(runtime.prediction.confidence * 100)}%`;
  const predictionText = runtime.prediction.message || runtime.prediction.label;
  const primaryReplayLabel = replayPaused ? t.continue : t.start;

  const latestRows = useMemo(() => {
    return Object.entries(runtime.latestData).slice(0, 10);
  }, [runtime.latestData]);

  useEffect(() => {
    let events: EventSource | null = null;

    async function boot() {
      try {
        const response = await fetch(`${API_BASE}/api/state`);
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        const data = (await response.json()) as StateResponse;
        setRuntime(parseRuntime(asRecord(data.state)));
        setWarnings(Array.isArray(data.warnings) ? data.warnings : []);
        setApiOnline(true);
      } catch (error) {
        setRuntime((current) => ({
          ...current,
          status: { message: `Dashboard API is not available: ${String(error)}`, kind: "danger" },
        }));
        setApiOnline(false);
      }

      events = new EventSource(`${API_BASE}/api/events`);
      events.onopen = () => setApiOnline(true);
      events.onerror = () => setApiOnline(false);
      events.onmessage = (message) => {
        const backendEvent = JSON.parse(message.data) as BackendEvent;
        handleBackendEvent(backendEvent);
      };
    }

    boot();
    return () => {
      events?.close();
    };
  }, []);

  function handleBackendEvent(backendEvent: BackendEvent) {
    const payload = asRecord(backendEvent.payload);

    if (backendEvent.event === "state_snapshot") {
      const nextRuntime = parseRuntime(payload);
      setRuntime(nextRuntime);
      addSpeedPoint(nextRuntime.speedLimit.gpsSpeed ?? getGpsSpeed(nextRuntime.latestData));
      return;
    }

    if (backendEvent.event === "warning_logged") {
      setWarnings((current) => [payload, ...current].slice(0, 20));
      return;
    }

    if (backendEvent.event === "warnings_reset") {
      setWarnings([]);
      setSpeedPoints([]);
      setReplayPaused(false);
    }
  }

  function addSpeedPoint(value: number | null) {
    if (value === null) {
      return;
    }
    setSpeedPoints((current) => [...current.slice(-23), value]);
  }

  async function post(path: string, body?: BodyInit, headers?: HeadersInit) {
    const response = await fetch(`${API_BASE}${path}`, { method: "POST", body, headers });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.json();
  }

  async function uploadCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      await post("/api/csv/upload", formData);
      setReplayPaused(false);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function switchLanguage() {
    await post("/api/language", JSON.stringify({ language: "en" }), { "Content-Type": "application/json" });
  }

  async function startReplay() {
    setReplayPaused(false);
    await post("/api/replay/start");
  }

  async function pauseReplay() {
    if (runtime.csv.simulating) {
      setReplayPaused(true);
    }
    await post("/api/replay/stop");
  }

  async function resetReplay() {
    setReplayPaused(false);
    await post("/api/replay/reset");
  }

  return (
    <main className="dashboard-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">
            <Car size={22} />
          </div>
          <div>
            <h1>{t.title}</h1>
            <p>{t.subtitle}</p>
          </div>
        </div>

        <button className="sidebar-button" onClick={switchLanguage}>
          <Languages size={16} />
          {t.language}
        </button>

        <SidebarItem
          icon={<CheckCircle2 size={18} />}
          label={t.model}
          value={runtime.modelApi.modelLoaded ? runtime.modelApi.modelFile : runtime.status.message}
          kind={runtime.modelApi.modelLoaded ? "success" : "warning"}
        />
        <SidebarItem
          icon={<FileSpreadsheet size={18} />}
          label={t.csvReplay}
          value={runtime.csv.simulating ? t.running : runtime.csv.selectedName || t.ready}
          kind={runtime.csv.simulating ? "info" : runtime.csv.selectedName ? "success" : "muted"}
        />
        <SidebarItem
          icon={<Activity size={18} />}
          label="API"
          value={apiOnline ? t.connected : t.disconnected}
          kind={apiOnline ? "success" : "danger"}
        />
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span>{t.selectedCsv}</span>
            <strong>{runtime.csv.selectedName || t.noCsv}</strong>
          </div>
          <div className={`status-pill ${classForKind(runtime.status.kind)}`}>{runtime.status.message}</div>
          <div className="actions">
            <input ref={fileInputRef} type="file" accept=".csv" onChange={uploadCsv} hidden />
            <button onClick={() => fileInputRef.current?.click()}>
              <Upload size={16} />
              {uploading ? "..." : t.upload}
            </button>
            <button className="primary" onClick={startReplay}>
              <Play size={16} />
              {primaryReplayLabel}
            </button>
            <button onClick={pauseReplay}>
              <Pause size={16} />
              {t.pause}
            </button>
            <button onClick={resetReplay}>
              <RotateCcw size={16} />
              {t.reset}
            </button>
          </div>
        </header>

        <section className="summary-grid">
          <article className={`prediction-panel ${classForKind(runtime.prediction.kind)}`}>
            <div className="panel-label">
              <Gauge size={20} />
              {t.prediction}
            </div>
            <h2 style={{ "--prediction-title-size": predictionTitleSize(predictionText) } as CSSProperties}>{predictionText}</h2>
            <div className="prediction-meta">
              <span>
                {t.confidence}: {confidence}
              </span>
              <span>
                {t.warningSource}: {runtime.prediction.warningSource || t.normal}
              </span>
            </div>
          </article>

          <MetricCard label={t.gpsSpeed} value={displayNumber(gpsSpeed, " km/h")} kind={runtime.prediction.kind} />
          <MetricCard
            label={t.speedLimit}
            value={displayNumber(runtime.speedLimit.limit, " km/h")}
            detail={runtime.speedLimit.message}
            kind={runtime.speedLimit.kind}
          />
          <MetricCard label={t.replayProgress} value={`${runtime.window.current} / ${runtime.window.total}`} detail={`${progress}%`} kind="info" />
        </section>

        <section className="detail-grid">
          <article className="panel warning-panel">
            <PanelTitle icon={<AlertTriangle size={18} />} title={t.warnings} value={String(warnings.length)} />
            {warnings.length ? (
              <div className="warning-list">
                {warnings.map((warning, index) => (
                  <div className="warning-row" key={`${textValue(warning.timestamp)}-${index}`}>
                    <strong>{textValue(warning.predicted_class, textValue(warning.warning_source, "-"))}</strong>
                    <span>{textValue(warning.timestamp, "-")}</span>
                    <small>
                      {displayNumber(numberValue(warning.gps_speed), " km/h")} - {textValue(warning.warning_source, "-")}
                    </small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-text">{t.noWarnings}</p>
            )}
          </article>

          <article className="panel chart-panel">
            <PanelTitle icon={<BarChart3 size={18} />} title={t.speedTrend} value={`${speedPoints.length}`} />
            <SpeedChart points={speedPoints} />
          </article>

          <article className="panel data-panel">
            <PanelTitle icon={<FileSpreadsheet size={18} />} title={t.latestData} value={`${latestRows.length}`} />
            {latestRows.length ? (
              <table>
                <tbody>
                  {latestRows.map(([key, value]) => (
                    <tr key={key}>
                      <th>{latestDataLabel(key)}</th>
                      <td>{latestDataValue(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="empty-text">{t.noData}</p>
            )}
          </article>
        </section>
      </section>
    </main>
  );
}

function SidebarItem({
  icon,
  label,
  value,
  kind,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  kind: StatusKind;
}) {
  return (
    <div className="sidebar-item">
      <div className="sidebar-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <i className={classForKind(kind)} />
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  kind,
}: {
  label: string;
  value: string;
  detail?: string;
  kind: StatusKind;
}) {
  return (
    <article className={`metric-card ${classForKind(kind)}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <p>{detail}</p> : null}
    </article>
  );
}

function PanelTitle({ icon, title, value }: { icon: ReactNode; title: string; value: string }) {
  return (
    <div className="panel-title">
      <div>
        {icon}
        <h3>{title}</h3>
      </div>
      <span>{value}</span>
    </div>
  );
}

function SpeedChart({ points }: { points: number[] }) {
  if (points.length < 2) {
    return <div className="chart-empty">Waiting for replay data</div>;
  }

  const width = 520;
  const height = 180;
  const max = Math.max(...points, 100);
  const min = Math.min(...points, 0);
  const range = Math.max(max - min, 1);
  const path = points
    .map((point, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - ((point - min) / range) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg className="speed-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="GPS speed trend">
      <path d={path} />
    </svg>
  );
}
