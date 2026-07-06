# Manual Driving Labeling System

This folder contains a local browser tool for manually labeling GPS driving windows.

The goal is simple:

1. Start the local site.
2. Select a CSV/XLSX file.
3. Label each generated driving window from a human perspective.
4. Export the manual labels as CSV.

This is better for thesis evidence than training directly on rule-generated labels, because the final labels come from human judgment.

## Supported Input

The uploaded file must be a CSV or XLSX file with:

- `vid_md5`
- one supported time column
- optional driving signals such as speed, heading, brake signal, turn signals, road ID, district, and vehicle state

The original `gps_1101.csv` can be used if it has the expected columns.

## Setup

Run this from the project root:

```powershell
python -m pip install -r requirements.txt
```

If your project virtual environment works, you can replace `python` with:

```powershell
.\.venv312\Scripts\python.exe
```

## Run The Manual Labeling Site

Start the server:

```powershell
python -m uvicorn llm_labeling_system.review_server:app --host 127.0.0.1 --port 8010
```

Open this URL:

```text
http://127.0.0.1:8010
```

## Labeling Workflow

1. Click **Select CSV/XLSX File** and choose your data file.
2. Set the window options:
   - `Window size`: number of rows in one labeling unit.
   - `Stride`: how far the next window moves forward.
   - `Max rows`: optional limit for testing with a small sample.
   - `Max windows`: maximum labeling units to generate.
3. Click **Start Labeling**.
4. For each window, inspect:
   - speed summary
   - heading change
   - brake count
   - turn signals
   - road and district context
   - raw rows
5. Optionally click **AI Suggest** to ask DeepSeek for a draft label.
6. Review the suggestion.
7. Choose the final label and save.

The AI suggestion does not save automatically. You must review it and click **Save And Next**.

To use **AI Suggest**, set your API key before starting the server:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
python -m uvicorn llm_labeling_system.review_server:app --host 127.0.0.1 --port 8010
```

## Labels

The system uses these labels:

- `normal`
- `speeding`
- `harsh_accel_brake`
- `zigzag_unstable`
- `unclear`

Use `unclear` when the data is not enough to make a confident judgment.

## Outputs

Manual labels are saved here:

```text
llm_labeling_system/outputs/manual_labels.jsonl
llm_labeling_system/outputs/manual_labels.csv
```

The browser export button downloads:

```text
manual_labels.csv
```

Temporary uploaded files are stored in:

```text
llm_labeling_system/uploads/
```

## Optional LLM Batch Mode

The older DeepSeek batch labeling code is still available for comparison or pre-labeling.

Mock test:

```powershell
python -m llm_labeling_system.batch_label --mock --limit 20
```

Real DeepSeek labeling requires:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
python -m llm_labeling_system.batch_label --limit 20
```

For thesis use, treat LLM labels as weak labels. Manually checked labels are stronger evidence.
