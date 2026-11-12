import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import MAX_FILE_MB
from data_prep import dataframe_profile, apply_plan, write_metadata_csv
from llm import get_plan
from ml_train import train_dataset

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="INFINITRIX Backend",
    description="Upload a tabular dataset, choose a target and task, then receive an AI-prepared CSV.",
    version="1.0.0",
)
# Local development only. Restrict origins before exposing this API publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"name": "INFINITRIX Backend", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/prepare")
async def prepare_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    model_type: str = Form(...),
):
    if model_type not in {"REGRESSION", "CLASSIFICATION"}:
        raise HTTPException(400, "model_type must be REGRESSION or CLASSIFICATION.")
    if not file.filename:
        raise HTTPException(400, "No filename was provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV, XLSX, and XLS files are supported.")

    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_FILE_MB} MB limit.")

    input_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    input_path.write_bytes(content)

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(input_path)
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding="latin-1")
        else:
            df = pd.read_excel(input_path)

        if df.empty:
            raise ValueError("The uploaded file has no data rows.")
        if len(df.columns) < 2:
            raise ValueError("The dataset must contain at least two columns.")
        if df.columns.duplicated().any():
            raise ValueError("The dataset contains duplicate column names. Rename them and upload again.")
        df.columns = [str(c).strip() for c in df.columns]
        if df.columns.duplicated().any():
            raise ValueError("Column names must be unique after trimming whitespace.")
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found. Available columns: {', '.join(df.columns)}"
            )

        profile = dataframe_profile(df)
        plan = get_plan(profile, target_column, model_type)
        cleaned, summary = apply_plan(df, target_column, plan)

        output_name = f"{Path(file.filename).stem}_cleaned_{uuid.uuid4().hex[:8]}.csv"
        output_path = OUTPUT_DIR / output_name
        write_metadata_csv(cleaned, target_column, model_type, str(output_path))

        return {
            "status": "success",
            "original_filename": file.filename,
            "cleaned_filename": output_name,
            "target_column": target_column,
            "model_type": model_type,
            "rows": int(cleaned.shape[0]),
            "columns": int(cleaned.shape[1]),
            "ai_provider": "OpenRouter",
            "ai_model": os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
            "ai_reasoning": str(plan.get("reasoning", "")),
            "cleaning_summary": summary,
            "download_url": f"/download/{output_name}",
        }
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    model_type: str = Form(...),
):
    """Train and compare general-purpose ML models on the cleaned CSV."""
    if model_type not in {"REGRESSION", "CLASSIFICATION"}:
        raise HTTPException(400, "model_type must be REGRESSION or CLASSIFICATION.")
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(400, "Training expects the cleaned CSV file.")
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_FILE_MB} MB limit.")
    try:
        from io import BytesIO
        try:
            df = pd.read_csv(BytesIO(content), skiprows=2)
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), skiprows=2, encoding="latin-1")
        if df.empty:
            raise ValueError("The cleaned dataset has no data rows.")
        df.columns = [str(c).strip() for c in df.columns]
        return train_dataset(df, target_column, model_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Model training failed: {str(exc)}") from exc

@app.get("/download/{filename}")
def download_file(filename: str):
    # Prevent path traversal; only serve files inside OUTPUT_DIR.
    safe_name = Path(filename).name
    path = OUTPUT_DIR / safe_name
    if not path.is_file():
        raise HTTPException(404, "Output file not found.")
    return FileResponse(path, media_type="text/csv", filename=safe_name)
