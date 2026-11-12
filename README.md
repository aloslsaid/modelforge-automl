# ModelForge AutoML

AI-assisted tabular-data preparation and model comparison using FastAPI, pandas, scikit-learn, and OpenRouter.

## Features
- CSV/XLSX/XLS upload with a configurable size limit.
- Dataset profiling and AI-generated conservative cleaning plans.
- Allow-listed transformations; model output is treated as untrusted data, never executable code.
- Target-column protection during preparation.
- Regression and classification model comparison with preprocessing pipelines.
- Held-out evaluation metrics and serialized model artifacts.

## Structure
- `main.py`: FastAPI application and API routes.
- `config.py`: environment-based configuration.
- `data_prep.py`: dataset profiling and safe transformations.
- `llm.py`: OpenRouter integration.
- `ml_train.py`: model candidates, preprocessing, evaluation, and model persistence.
- `schemas.py`: request and response schemas.
- `uploads/`, `outputs/`, `models/`: runtime storage.

## Requirements and installation
Python 3.10+ and the packages in `requirements.txt`.

```bash
git clone https://github.com/aloslsaid/modelforge-automl.git
cd modelforge-automl
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file and set `OPENROUTER_API_KEY`. Never commit real credentials. Optional settings include `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, and `MAX_FILE_MB`.

## Run

From the repository root:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`

## API

### `POST /prepare`
Multipart fields: `file` (CSV/XLSX/XLS), `target_column` (exact column name), and `model_type` (`REGRESSION` or `CLASSIFICATION`). Returns a preparation summary and a download URL.

### `GET /download/{filename}`
Downloads a generated prepared CSV.

### `POST /train`
Multipart fields: `file` (prepared CSV), `target_column`, and `model_type`. Returns up to five model comparisons, metrics, and a generated model identifier.

## Evaluation
Regression metrics: R², MAE, RMSE. Classification metrics: accuracy, weighted precision, weighted recall, weighted F1. Numeric and categorical preprocessing is fitted inside scikit-learn pipelines. The target is excluded from the feature matrix. Results depend on the data and split; they are not guarantees of future performance.

## Data safety and security
- The AI receives a profile/sample of uploaded data to recommend transformations. Do not upload confidential or regulated data unless authorized and comfortable with the provider's data policy.
- The model returns a JSON plan, not Python code. The backend applies only supported operations and validates column names.
- Keep credentials in environment variables or an untracked `.env`.
- The development server is intended for localhost. Before public deployment, add authentication, rate limiting, strict CORS, storage quotas, retention controls, and access restrictions.
- Treat uploaded files, model outputs, and serialized artifacts as untrusted. Do not load joblib files from unknown sources.

## Limitations
AI cleaning is heuristic and requires review. Model search may be expensive. The default random split is not appropriate for every time-series or grouped dataset; use domain-specific splitting where needed. No production readiness or predictive performance is guaranteed.

## License
No license file is currently included; add a license explicitly if you intend to grant reuse rights.
