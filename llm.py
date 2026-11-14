import json
import httpx
from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

SYSTEM_PROMPT = """You are a careful tabular-data preparation assistant.
Return ONLY valid JSON. Do not return markdown or code.
You must not invent columns or values. Use only the provided column names.
Recommend conservative, safe cleaning and feature engineering actions.
The target column must NEVER be modified, filled, encoded, scaled, or used to create features.
Do not drop rows. Do not remove outliers automatically.
Allowed actions:
- trim_whitespace: list of column names; trim leading/trailing whitespace in string cells.
- normalize_empty_strings: list of column names; convert blank strings to missing.
- fill_numeric_median: list of non-target numeric column names.
- fill_categorical_mode: list of non-target categorical column names.
- drop_constant_columns: list of non-target column names that are constant.
- datetime_columns: list of non-target columns that clearly contain dates.
- create_datetime_parts: list of names from datetime_columns; create year/month/day features.
- encode_categoricals: list of non-target categorical columns; one-hot encode them.
Return this exact shape:
{
 "reasoning": "short explanation",
 "actions": {
  "trim_whitespace": [],
  "normalize_empty_strings": [],
  "fill_numeric_median": [],
  "fill_categorical_mode": [],
  "drop_constant_columns": [],
  "datetime_columns": [],
  "create_datetime_parts": [],
  "encode_categoricals": []
 }
}
Only include columns that exist in the dataset. Keep actions conservative.
"""

def get_plan(profile: dict, target: str, model_type: str) -> dict:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "PASTE_YOUR_OPENROUTER_KEY_HERE":
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to backend/.env.")
    user_payload = {
        "target_column": target,
        "model_type": model_type,
        "dataset_profile": profile,
        "instruction": "Create a safe cleaning and feature-engineering plan. Do not write code."
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "INFINITRIX Data Preparation Backend",
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ],
        "temperature": 0.1,
        "max_tokens": 1600,
    }
    try:
        response = httpx.post(OPENROUTER_BASE_URL, headers=headers, json=body, timeout=90)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1200]
        raise RuntimeError(f"OpenRouter HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not connect to OpenRouter: {exc}") from exc

    try:
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = str(content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Model did not return a JSON object.")
        plan = json.loads(text[start:end + 1])
        if not isinstance(plan, dict) or not isinstance(plan.get("actions"), dict):
            raise ValueError("AI plan has an unexpected structure.")
        return plan
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse AI response as a valid cleaning plan: {exc}") from exc
