# ModelForge AutoML — AI-Assisted Tabular Machine Learning

> A FastAPI-based AutoML backend that profiles datasets, proposes safe preparation steps with an LLM, and compares machine-learning models for regression or classification.

## Overview
ModelForge is built to automate repetitive parts of a tabular ML workflow. It accepts a dataset, inspects its structure, asks an OpenRouter model to propose a constrained preparation plan, validates that plan in Python, and trains candidate estimators for the task selected by the user.

## Workflow
1. Upload CSV, XLSX, or XLS data.
2. Profile rows, columns, types, missing values, unique values, and representative samples.
3. Select a target column and choose `REGRESSION` or `CLASSIFICATION`.
4. Generate a structured AI cleaning/feature-engineering plan.
5. Validate all proposed actions against an allowlist and the actual dataset columns.
6. Apply safe transformations locally while keeping the target column unchanged.
7. Train and compare candidate models using cross-validation and hyperparameter search.
8. Evaluate the strongest candidates on a held-out test split.
9. Save the selected fitted pipeline and metadata as a Joblib artifact.

## AI-assisted preparation
Supported operations include trimming whitespace, normalizing blank strings, median imputation for numeric features, mode imputation for categorical features, identifying constant columns, detecting date columns, creating calendar components, and encoding categorical values. The AI proposes actions; deterministic application code validates column names and prevents the target from being modified.

## Model evaluation
**Regression:** R², MAE, and RMSE.

**Classification:** Accuracy, weighted Precision, weighted Recall, weighted F1, and binary decision-threshold optimization where supported.

The project description advertises comparisons across 15 regression and 15 classification algorithms. Model availability and successful fitting depend on the dataset's size, data types, class distribution, and estimator constraints.

## Stack
Python · FastAPI · Pandas · NumPy · Scikit-learn · HTTPX · OpenRouter · Pydantic · Joblib · openpyxl

## Install
```bash
git clone https://github.com/aloslsaid/modelforge-automl.git
cd modelforge-automl
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env`:
```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_MODEL=nvidia/nemotron-3.5-super-vl:free
MAX_FILE_MB=30
MAX_ROWS_FOR_AI_SAMPLE=20
```
Never commit real API credentials. The model can be changed to one available to your OpenRouter account.

## API endpoints
- `GET /` — service information.
- `GET /health` — health check.
- `POST /prepare` — upload and prepare a dataset.
- `POST /train` — train and compare models.
- `GET /download/{filename}` — download generated files.

The interactive API documentation is available at `http://127.0.0.1:8000/docs` after successful startup.

## Launch command and source-layout caveat
Intended command:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
The current root files are flat, but `main.py` imports modules through `app.*` paths. The package layout must match those imports for startup to succeed. This documentation update does not alter the source or claim that a fresh runtime test has passed.

## ML reliability
Automated model selection is not a guarantee of generalization. Inspect data leakage, class imbalance, target quality, validation design, and performance on truly unseen data before deployment.

## Skills demonstrated
AutoML · LLM-assisted preparation · Feature engineering · Model selection · Cross-validation · Hyperparameter search · FastAPI · Model persistence
