# INFINITRIX Backend — AI CSV Preparation

Backend only. No frontend. FastAPI provides interactive testing at `/docs`.

## What it does
1. Accepts CSV, XLSX, or XLS.
2. Reads the dataset and profiles columns, types, missing values, unique values, and sample rows.
3. You provide `target_column` and `model_type` (`REGRESSION` or `CLASSIFICATION`).
4. Sends a compact profile/sample to OpenRouter using the configured model.
5. The AI returns a constrained JSON cleaning/feature-engineering plan.
6. Python validates the plan and executes only allowlisted operations locally.
7. Produces a downloadable CSV with two metadata lines before the regular CSV header:
   - `# wanted_column="..."`
   - `# model_type=REGRESSION` or `# model_type=CLASSIFICATION`

The target column is protected from cleaning, encoding, scaling, and feature creation. The AI never receives permission to execute Python code.

## Setup (Windows PowerShell)

```powershell
cd path\to\infinitrix_backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Put your OpenRouter key after `OPENROUTER_API_KEY=` and save `.env`.
Do not put the key in Python source code or upload `.env` to GitHub.

## Run

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Test
- Expand `POST /prepare`.
- Choose a CSV/XLSX file.
- Enter the exact target column name, e.g. `Churn`.
- Enter `CLASSIFICATION` for categories/classes or `REGRESSION` for numeric values.
- Execute. The response contains a `download_url`.
- Open `http://127.0.0.1:8000` + the returned download URL to download the prepared CSV.

## Important CSV format
The output intentionally starts with two comment-style metadata rows. The future ML `main.py` should read them with:

```python
with open("dataset_cleaned.csv", encoding="utf-8-sig") as f:
    wanted_column = f.readline().strip().split("=", 1)[1]
    model_type = f.readline().strip().split("=", 1)[1]
import pandas as pd
df = pd.read_csv("dataset_cleaned.csv", skiprows=2)
```

This means the CSV is not a plain CSV from line 1; always use `skiprows=2` when loading it for ML.

## Notes
- OpenRouter free models can be rate-limited or unavailable temporarily.
- AI output is treated as an untrusted plan and validated before execution.
- This initial version does not train a model. It only prepares the dataset and writes the metadata required by the next stage.
- For production, add authentication, file retention limits, job queueing, and stronger CSV/XLSX validation.
