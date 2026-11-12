from fastapi.testclient import TestClient

import main
from data_prep import apply_plan


def test_prepare_rejects_oversized_upload_before_ai(monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_MB", 1)
    monkeypatch.setattr(main, "get_plan", lambda *args: (_ for _ in ()).throw(AssertionError("AI must not run")))
    client = TestClient(main.app)
    response = client.post(
        "/prepare",
        data={"target_column": "label", "model_type": "REGRESSION"},
        files={"file": ("data.csv", b"x" * (1024 * 1024 + 1), "text/csv")},
    )
    assert response.status_code == 413


def test_train_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_MB", 1)
    client = TestClient(main.app)
    response = client.post(
        "/train",
        data={"target_column": "label", "model_type": "REGRESSION"},
        files={"file": ("data.csv", b"x" * (1024 * 1024 + 1), "text/csv")},
    )
    assert response.status_code == 413


def test_ai_plan_cannot_change_or_drop_target_column():
    import pandas as pd

    source = pd.DataFrame({"feature": [" a ", "b"], "target": ["yes", "no"]})
    plan = {
        "actions": {
            "trim_whitespace": ["feature", "target"],
            "drop_constant_columns": ["target"],
            "encode_categoricals": ["target"],
            "fill_categorical_mode": ["target"],
            "invented_action": ["feature"],
        }
    }
    cleaned, _ = apply_plan(source, "target", plan)
    assert cleaned["target"].tolist() == source["target"].tolist()
    assert "target" in cleaned.columns
    assert cleaned["feature"].tolist() == ["a", "b"]
