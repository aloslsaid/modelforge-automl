# ModelForge AutoML

ModelForge is an experimental tabular-data preparation and model-comparison application. It combines an OpenRouter-assisted cleaning plan with deterministic transformations, then trains and compares scikit-learn regression or classification pipelines.

## Features

- CSV, XLSX and XLS upload support.
- Dataset profiling and target-column selection.
- AI-proposed, allow-listed cleaning and feature-engineering actions.
- Explicit protection of the selected target column during preparation.
- Regression and classification model search with preprocessing pipelines.
- Held-out test metrics and serialized best-model artifacts.

## Architecture

The implementation is kept at the repository root. The `app/` package contains compatibility entry points so the documented ASGI command works without moving or duplicating the original modules.

- `main.py`: FastAPI endpoints (`/health`, `/prepare`, `/train`, `/download/{filename}`).
- `data_prep.py`: profiling, validated transformations and CSV export.
- `llm.py`: OpenRouter request and JSON-plan parsing.
- `ml_train.py`: scikit-learn preprocessing, model selection and serialization.
- `config.py`: environment-based settings.
- `code.html`: standalone browser interface.

## Requirements and setup

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set a valid OpenRouter key:

```bash
# Windows PowerShell
Copy-Item .env.example .env
# macOS/Linux
cp .env.example .env
``

Edit `.env` and set `OPENROUTER_API_KEY`. Never commit the real file.

Start the API from the repository root:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`. Open `code.html` in a browser for the standalone UI; it can be configured to use the local API address.

## API workflow

1. Upload a tabular file and choose its target column and task.
2. The backend profiles the data and requests a conservative JSON plan from OpenRouter.
3. The backend validates proposed column names and applies only supported transformations.
4. Download the cleaned CSV. It contains two metadata comment lines before the CSV header.
5. Optionally submit that cleaned file to `/train` for model comparison.

Training reads the exported format with `skiprows=2`. Model artifacts are stored locally in `models/`; uploaded and generated files are stored in `uploads/` and `outputs/`.

## Evaluation and limitations

Metrics are dataset- and split-dependent. The reported test metrics are not a guarantee of future performance. Inspect class balance, target quality, leakage risks, temporal structure and data provenance before trusting a result. The current search evaluates multiple candidate models and can be computationally expensive on large datasets.

AI-generated plans are suggestions, not ground truth. Review the cleaning summary and retain the original dataset. Do not upload sensitive, regulated or confidential data to an external provider without authorization.

## Tests

Run the focused data-preparation tests from the repository root:

```bash
python -m pytest
```

A successful test run verifies only the covered behavior; it does not validate every model, external API, or deployment configuration.

## Security

- Keep API keys in an untracked `.env` file.
- The API is intended for local development; add authentication, rate limits, storage cleanup and restrictive CORS before public deployment.
- Do not expose generated model artifacts or uploaded datasets publicly.
- Treat uploaded files and model-generated plans as untrusted input.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
