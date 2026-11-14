import json
import pandas as pd

ACTION_KEYS = (
    "trim_whitespace", "normalize_empty_strings", "fill_numeric_median",
    "fill_categorical_mode", "drop_constant_columns", "datetime_columns",
    "create_datetime_parts", "encode_categoricals",
)

def dataframe_profile(df: pd.DataFrame) -> dict:
    sample = df.head(20).replace({pd.NA: None}).to_dict(orient="records")
    columns = []
    for name in df.columns:
        s = df[name]
        columns.append({
            "name": str(name),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "examples": [str(v)[:100] for v in s.dropna().head(4).tolist()],
        })
    return {
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "sample_rows": sample,
    }

def _safe_list(actions: dict, key: str, valid_columns: set, target: str) -> list[str]:
    values = actions.get(key, [])
    if not isinstance(values, list):
        return []
    return [v for v in values if isinstance(v, str) and v in valid_columns and v != target]

def apply_plan(df: pd.DataFrame, target: str, raw_plan: dict) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    actions = raw_plan.get("actions", {})
    valid = set(map(str, result.columns))
    summary: list[str] = []

    # Always preserve the target exactly as uploaded.
    if target not in valid:
        raise ValueError(f"Target column '{target}' does not exist.")

    for col in _safe_list(actions, "trim_whitespace", valid, target):
        if result[col].dtype == "object" or pd.api.types.is_string_dtype(result[col]):
            result[col] = result[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            summary.append(f"Trimmed whitespace in '{col}'.")

    for col in _safe_list(actions, "normalize_empty_strings", valid, target):
        result[col] = result[col].replace(r"^\s*$", pd.NA, regex=True)
        summary.append(f"Converted blank strings to missing values in '{col}'.")

    for col in _safe_list(actions, "fill_numeric_median", valid, target):
        if pd.api.types.is_numeric_dtype(result[col]):
            median = result[col].median()
            if pd.notna(median):
                result[col] = result[col].fillna(median)
                summary.append(f"Filled missing numeric values in '{col}' with its median.")

    for col in _safe_list(actions, "fill_categorical_mode", valid, target):
        if not pd.api.types.is_numeric_dtype(result[col]):
            modes = result[col].mode(dropna=True)
            if not modes.empty:
                result[col] = result[col].fillna(modes.iloc[0])
                summary.append(f"Filled missing categorical values in '{col}' with its mode.")

    # Date parsing and derived calendar columns.
    datetime_cols = _safe_list(actions, "datetime_columns", valid, target)
    create_parts = set(_safe_list(actions, "create_datetime_parts", valid, target))
    for col in datetime_cols:
        parsed = pd.to_datetime(result[col], errors="coerce")
        if parsed.notna().sum() > 0:
            result[col] = parsed
            if col in create_parts:
                for part in ("year", "month", "day"):
                    new_name = f"{col}_{part}"
                    if new_name not in result.columns:
                        result[new_name] = getattr(parsed.dt, part)
                summary.append(f"Created year/month/day features from '{col}'.")
            summary.append(f"Parsed '{col}' as datetime.")

    # Drop only columns explicitly identified as constant, never target.
    for col in _safe_list(actions, "drop_constant_columns", set(map(str, result.columns)), target):
        if result[col].nunique(dropna=False) <= 1:
            result = result.drop(columns=[col])
            summary.append(f"Dropped constant column '{col}'.")

    # One-hot encode only explicitly approved, non-target categoricals.
    encode_cols = _safe_list(actions, "encode_categoricals", set(map(str, result.columns)), target)
    encode_cols = [c for c in encode_cols if not pd.api.types.is_numeric_dtype(result[c])]
    if encode_cols:
        result = pd.get_dummies(result, columns=encode_cols, dummy_na=True, dtype=int)
        summary.append(f"One-hot encoded categorical columns: {', '.join(encode_cols)}.")

    # Restore target values/order exactly; no AI operation is allowed to mutate it.
    result[target] = df[target].values
    if not summary:
        summary.append("No safe transformations were selected by the AI; original data retained.")
    return result, summary

def write_metadata_csv(df: pd.DataFrame, target: str, model_type: str, output_path: str):
    # Two metadata lines at the top, followed by the normal CSV header and data.
    # These first two lines are intentionally comments for the future main.py reader.
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(f"# wanted_column={json.dumps(target, ensure_ascii=False)}\n")
        f.write(f"# model_type={model_type}\n")
        df.to_csv(f, index=False)
