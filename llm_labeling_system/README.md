# Manual Driving Labeling Platform

A local, browser-based tool for manually labeling GPS driving windows from a
human perspective. Human-reviewed labels are stronger thesis evidence than
rule-generated or LLM-generated labels.

Data flows: **upload a GPS file → slice per-vehicle trajectory windows →
label each window → export CSV/JSONL**. Labels are stored in a local SQLite
database, scoped per session, with an optional DeepSeek "AI Suggest" draft.

## Highlights

- **Sessions** — each uploaded dataset is its own session; labels never leak
  between files. Switch between sessions and resume where you left off.
- **SQLite source of truth** with CSV/JSONL export in the original column format.
- **Dark mode** (system-aware + manual toggle, remembered per browser) and
  keyboard-accessible controls.
- **AI Suggest** — optional DeepSeek draft label, or an offline deterministic
  mock that needs no API key.

## Architecture

```
llm_labeling_system/
  app.py            FastAPI app factory (mounts routers + static)
  config.py         Settings from environment (model, data dir, upload caps)
  db.py             SQLite connection (WAL) + schema
  repository.py     Data-access layer (projects / sessions / windows / labels)
  schemas.py        Pydantic request/response models
  routers/          sessions, windows, labels, suggest, meta
  services/         windowing, prompting, deepseek_client, export
  batch_label.py    CLI: batch-label into the DB (DeepSeek or mock)
  import_legacy.py  One-time importer for old outputs/*.jsonl label files
  static/           index.html + css/ (tokens, app) + js/ (ES modules)
  tests/            pytest: windowing, repository, API round-trip
  data/             SQLite db + uploads + exports (gitignored)
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

Open <http://127.0.0.1:8010>.

## Labeling workflow

1. **Create Session** — choose a CSV/XLSX file and set window options:
   - `Window size`: rows per labeling unit.
   - `Stride`: how far the next window advances.
   - `Max rows` / `Max windows`: optional limits for quick sampling.
   - `Project` (optional): a name to group related sessions.
2. Pick the **Active session** from the dropdown (uploading selects it for you).
3. For each window, inspect speed summary, heading change, brake/turn signals,
   road/district context, data-quality flags, and the raw rows.
4. Optionally click **AI Suggest** for a draft (see below), then review it.
5. Choose the final label, set confidence, and click **Save And Next**.

The AI suggestion never saves automatically — you must review and save.

### Supported input

CSV or XLSX with a `vid_md5` column, one time column (e.g. `gps时间`), and
optional signals (speed, heading, brake, turn signals, road ID, district,
vehicle state). The original `gps_1101.csv` works if it has these columns.

### Labels

`normal`, `speeding`, `harsh_accel_brake`, `zigzag_unstable`, `unclear`.
Use `unclear` when the evidence is not enough for a confident judgment.

## AI Suggest (optional DeepSeek)

Without an API key, the app defaults to the **offline mock** so AI Suggest still
works. To use the real model, set the key before starting the server:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
python -m uvicorn llm_labeling_system.app:app --host 127.0.0.1 --port 8010
```

Configuration via environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | Enables the real DeepSeek suggestion path |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model id for suggestions |
| `LABELING_DATA_DIR` | `llm_labeling_system/data` | SQLite db + uploads + exports |
| `LABELING_MAX_UPLOAD_MB` | `200` | Hard cap on upload size |
| `LABELING_MAX_ROWS` | — | Optional global row cap when parsing |

## Exports

Use **Export CSV** in the toolbar, or call the API directly:

```text
GET /api/sessions/{id}/export?fmt=csv      # or fmt=jsonl
```

Exports are written to `data/exports/` and keep the original column contract
(`window_id, vehicle_id, start_time, end_time, label, confidence, …`).

## Batch LLM labeling (optional)

Treat LLM labels as **weak labels**; manually reviewed labels are stronger.

```powershell
# Offline mock (no key needed)
python -m llm_labeling_system.batch_label --input gps_1101.csv --mock --limit 20

# Real DeepSeek
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
python -m llm_labeling_system.batch_label --input gps_1101.csv --limit 20
```

Batch runs create/resume a session under the `batch` project and export
CSV/JSONL to `data/exports/`.

## Migrating old labels

If you have labels from the previous file-based version, import them once:

```powershell
python -m llm_labeling_system.import_legacy llm_labeling_system/outputs/manual_labels.jsonl
```

## Tests

```powershell
.\.venv312\Scripts\python.exe -m pytest llm_labeling_system/tests -q
```

Covers windowing math, repository session-scoping/upsert, and a full API
round-trip (upload → label → export → delete).
