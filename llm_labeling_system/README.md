# Manual Driving Labeling Platform

A local, browser-based tool for labeling GPS driving windows from a human
perspective, with an AI assistant that pre-screens a whole session so you
only have to review the windows that actually look abnormal. Human-reviewed
labels are stronger thesis evidence than rule-generated or LLM-generated labels.

Data flows: **upload a GPS file → slice per-vehicle trajectory windows →
optionally AI-scan the session → review the flagged windows → label → export
CSV/JSONL**. Labels are stored in a local SQLite database, scoped per session.

A dashboard screenshot is in the [repository root README](../README.md).

## Highlights

- **Sessions** — each uploaded dataset is its own session; labels never leak
  between files. Switch between sessions and resume where you left off.
- **AI Scan** — one click labels every window in a session in the background
  (with live progress) using the configured LLM, or an offline deterministic
  mock that needs no API key. The results panel lists only the windows flagged
  as abnormal or "unclear"; clicking one jumps straight to it, pre-filled with
  the AI's suggestion for you to accept or correct.
- **AI Settings dialog** — configure the API base URL, model, and key for any
  OpenAI-compatible provider (DeepSeek, OpenAI, local Ollama, …) at runtime,
  with a "Test connection" check. The key is stored server-side in SQLite and
  never sent back to the browser; environment variables remain a fallback.
- **SQLite source of truth** with two independent CSV/JSONL exports — human
  labels and raw AI-scan verdicts — both in the original column format.
- **Dashboard UI** — light/dark theming (system-aware + manual toggle,
  remembered per browser), card-based panels, and keyboard-accessible controls.

## Architecture

```
llm_labeling_system/
  app.py              FastAPI app factory (mounts routers + static)
  config.py           Settings from environment (model, data dir, upload caps)
  db.py               SQLite connection (WAL) + schema
  repository.py       Data-access layer (projects / sessions / windows / labels / ai_suggestions / app_settings)
  schemas.py          Pydantic request/response models
  routers/            sessions, windows, labels, suggest, scan, settings, meta
  services/
    windowing.py       CSV/XLSX parsing, per-vehicle window slicing, summaries
    prompting.py        System prompt, response validation, offline mock labeler
    deepseek_client.py  OpenAI-compatible chat client (label + test-connection calls)
    ai_settings.py      Resolves the effective AI config: DB settings override env vars
    scan_jobs.py        Background thread that AI-scans every window in a session
    export.py            CSV/JSONL export for human labels or AI-scan verdicts
  batch_label.py       CLI: batch-label into the DB (DeepSeek or mock)
  import_legacy.py     One-time importer for old outputs/*.jsonl label files
  static/               index.html + css/ (tokens, app) + js/ (ES modules)
  tests/                pytest: windowing, repository, API round-trip, settings, scan
  data/                 SQLite db + uploads + exports (gitignored)
```

## Setup

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Or use the project virtual environment:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
python -m uvicorn llm_labeling_system.app:app --host 127.0.0.1 --port 8010
```

Open <http://127.0.0.1:8010>. No API key is required to start — AI features
fall back to the offline mock until you configure one (see below).

## Labeling workflow

1. **Create Session** — choose a CSV/XLSX file and set window options:
   - `Window size`: rows per labeling unit.
   - `Stride`: how far the next window advances.
   - `Max rows` / `Max windows`: optional limits for quick sampling.
   - `Project` (optional): a name to group related sessions.
2. Pick the **Active session** from the dropdown (uploading selects it for you).
3. Click **Run AI Scan** to have the AI examine every window in the background;
   watch the progress bar, or **Cancel** at any time. Results persist even if
   you switch sessions or reload the page.
4. Work through the flagged rows in the results table — each click jumps to
   that window and pre-fills the label form with the AI's suggestion.
5. For each window, inspect speed summary, heading change, brake/turn signals,
   road/district context, data-quality flags, and the raw rows.
6. Choose the final label, set confidence, and click **Save And Next**.

AI suggestions never save automatically — you must review and save.

### Supported input

CSV or XLSX with a `vid_md5` column, one time column (e.g. `gps时间`), and
optional signals (speed, heading, brake, turn signals, road ID, district,
vehicle state). The original `gps_1101.csv` works if it has these columns.

### Labels

`normal`, `speeding`, `harsh_accel_brake`, `zigzag_unstable`, `unclear`.
Use `unclear` when the evidence is not enough for a confident judgment.

## AI assistance (optional, any OpenAI-compatible provider)

Without an API key, AI Scan defaults to the **offline mock** labeler so the
whole workflow still works end to end. To use a real model, open
**⚙ Settings** in the toolbar and enter the base URL, model name, and API
key — no restart needed. Use **Test connection** to verify before saving.

The key is written to the local SQLite database (`app_settings` table) and is
never echoed back to the browser; only a "configured" flag is exposed.

Environment variables remain a fallback if you'd rather configure it before
first launch, or for headless/CLI use (`batch_label.py`):

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
python -m uvicorn llm_labeling_system.app:app --host 127.0.0.1 --port 8010
```

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | Fallback API key, used only if none is saved in Settings |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Fallback model id |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Fallback API base URL |
| `LABELING_DATA_DIR` | `llm_labeling_system/data` | SQLite db + uploads + exports |
| `LABELING_MAX_UPLOAD_MB` | `200` | Hard cap on upload size |
| `LABELING_MAX_ROWS` | — | Optional global row cap when parsing |

## Exports

Two independent exports, both in the original column format
(`window_id, vehicle_id, start_time, end_time, label, confidence, …`), written
to `data/exports/`:

| Toolbar button | Endpoint | Source |
|---|---|---|
| **Export CSV** | `GET /api/sessions/{id}/export?fmt=csv` | Human-saved labels |
| **Export AI CSV** (in the AI Scan panel) | `GET /api/sessions/{id}/export?fmt=csv&source=ai` | Raw AI-scan verdicts (every scanned window, `mock` or `ai` sourced) |

Add `fmt=jsonl` to either for JSONL instead of CSV.

## Batch LLM labeling (optional CLI)

Treat LLM labels as **weak labels**; manually reviewed labels are stronger.

```powershell
# Offline mock (no key needed)
python -m llm_labeling_system.batch_label --input gps_1101.csv --mock --limit 20

# Real DeepSeek
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
python -m llm_labeling_system.batch_label --input gps_1101.csv --limit 20
```

Batch runs create/resume a session under the `batch` project and export
CSV/JSONL to `data/exports/`. This writes straight into the human `labels`
table (unlike AI Scan, which keeps its verdicts separate) — use it only for
offline dataset prep, not alongside interactive labeling of the same session.

## Migrating old labels

If you have labels from the previous file-based version, import them once:

```powershell
python -m llm_labeling_system.import_legacy llm_labeling_system/outputs/manual_labels.jsonl
```

## Tests

```powershell
.\.venv312\Scripts\python.exe -m pytest llm_labeling_system/tests -q
```

Covers windowing math, repository session-scoping/upsert, a full API
round-trip (upload → label → export → delete), AI settings persistence and
precedence, and the AI Scan job lifecycle (mock/real path, cancel, error
handling, export).
