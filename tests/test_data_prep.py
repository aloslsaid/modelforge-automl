import pandas as pd

from data_prep import apply_plan


def test_target_column_is_preserved_exactly():
    source = pd.DataFrame({"feature": [" a ", " b "], "target": [1, 0]})
    plan = {"actions": {"trim_whitespace": ["feature", "target"], "fill_numeric_median": ["target"]}}
    cleaned, _ = apply_plan(source, "target", plan)
    assert cleaned["target"].tolist() == source["target"].tolist()
    assert cleaned["feature"].tolist() == ["a", "b"]


def test_untrusted_ai_column_names_are_ignored():
    source = pd.DataFrame({"feature": ["x", "y"], "target": [1, 0]})
    plan = {"actions": {"drop_constant_columns": ["target", "missing_column"]}}
    cleaned, _ = apply_plan(source, "target", plan)
    assert "target" in cleaned.columns
    assert "missing_column" not in cleaned.columns
