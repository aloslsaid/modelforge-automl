from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TOP_N = 5
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.10, 0.901, 0.05)]


def _json_number(value: Any):
    value = float(value)
    return value if np.isfinite(value) else None


class ThresholdClassifier:
    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold

    @property
    def classes_(self):
        return self.pipeline.classes_

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[:, 1]
        classes = np.asarray(self.pipeline.classes_)
        return classes[(proba >= self.threshold).astype(int)]


def _native(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_native(v) for v in value]
    return value


def _clean_params(params: dict) -> dict:
    return {k.replace("model__", "", 1): _native(v) for k, v in params.items()}


def _prefix(grid):
    if isinstance(grid, list):
        return [_prefix(g) for g in grid]
    return {f"model__{k}": v for k, v in grid.items()}


def _search(estimator, grid, prep, X_train, y_train, scoring, cv):
    pipe = Pipeline([("preprocess", prep), ("model", estimator)])
    gs = GridSearchCV(pipe, _prefix(grid), scoring=scoring, cv=cv, n_jobs=-1, refit=True, error_score="raise")
    gs.fit(X_train, y_train)
    return gs


def _regression_candidates():
    tree_depth = [None, 10, 20]
    return [
        ("Ridge Regression", Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        ("Lasso", Lasso(max_iter=10000), {"alpha": [0.0005, 0.001, 0.01, 0.1, 1]}),
        ("ElasticNet", ElasticNet(max_iter=10000), {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}),
        ("Bayesian Ridge", BayesianRidge(), {"alpha_1": [1e-6, 1e-5], "alpha_2": [1e-6, 1e-5], "lambda_1": [1e-6, 1e-5], "lambda_2": [1e-6, 1e-5]}),
        ("Huber Regressor", HuberRegressor(max_iter=1000), {"epsilon": [1.2, 1.35, 1.5, 1.75], "alpha": [1e-4, 1e-3, 1e-2]}),
        ("K-Nearest Neighbors", KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}),
        ("Support Vector Regression", SVR(kernel="rbf"), {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5], "gamma": ["scale", "auto"]}),
        ("Decision Tree", DecisionTreeRegressor(random_state=42), {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 3, 5, 10]}),
        ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42), {"max_depth": tree_depth, "min_samples_leaf": [1, 3, 5], "max_features": [1.0, "sqrt"]}),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=300, random_state=42), {"max_depth": tree_depth, "min_samples_leaf": [1, 3, 5], "max_features": [1.0, "sqrt"]}),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42), {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 5], "subsample": [0.8, 1.0]}),
        ("Hist Gradient Boosting", HistGradientBoostingRegressor(max_iter=200, random_state=42), {"learning_rate": [0.03, 0.1, 0.2], "max_depth": [None, 6], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}),
        ("AdaBoost", AdaBoostRegressor(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.03, 0.1, 0.5, 1.0], "loss": ["linear", "square"]}),
        ("Bagging", BaggingRegressor(random_state=42), {"n_estimators": [50, 100], "max_samples": [0.6, 0.8, 1.0], "max_features": [0.6, 0.8, 1.0]}),
        ("MLP Neural Network", MLPRegressor(max_iter=600, early_stopping=True, random_state=42), {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 1e-2]}),
    ]


def _classification_candidates():
    tree_depth = [None, 10, 20]
    cw = [None, "balanced"]
    return [
        ("Logistic Regression", LogisticRegression(max_iter=3000), {"C": [0.01, 0.1, 1, 10, 100], "class_weight": cw}),
        ("K-Nearest Neighbors", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}),
        ("Support Vector Machine", SVC(kernel="rbf", probability=True, random_state=42), {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "class_weight": cw}),
        ("Decision Tree", DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 3, 5, 10], "class_weight": cw}),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42), {"max_depth": tree_depth, "min_samples_leaf": [1, 3, 5], "class_weight": cw}),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=300, random_state=42), {"max_depth": tree_depth, "min_samples_leaf": [1, 3, 5], "class_weight": cw}),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42), {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 5], "subsample": [0.8, 1.0]}),
        ("Hist Gradient Boosting", HistGradientBoostingClassifier(max_iter=200, random_state=42), {"learning_rate": [0.03, 0.1, 0.2], "max_depth": [None, 6], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}),
        ("AdaBoost", AdaBoostClassifier(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.03, 0.1, 0.5, 1.0]}),
        ("Bagging", BaggingClassifier(random_state=42), {"n_estimators": [50, 100], "max_samples": [0.6, 0.8, 1.0], "max_features": [0.6, 0.8, 1.0]}),
        ("Gaussian Naive Bayes", GaussianNB(), {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(), [{"solver": ["svd"]}, {"solver": ["lsqr"], "shrinkage": [None, "auto", 0.1, 0.5]}]),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis(), {"reg_param": [0.0, 0.05, 0.1, 0.3, 0.5]}),
        ("MLP Neural Network", MLPClassifier(max_iter=600, early_stopping=True, random_state=42), {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 1e-2]}),
        ("SGD Logistic", SGDClassifier(loss="log_loss", max_iter=3000, random_state=42), {"alpha": [1e-5, 1e-4, 1e-3, 1e-2], "penalty": ["l2", "elasticnet"], "class_weight": cw}),
    ]


def train_dataset(df: pd.DataFrame, target: str, model_type: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the cleaned dataset.")
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for model training.")

    # Drop rows with missing target only; target is never imputed or encoded as a feature.
    data = df.loc[df[target].notna()].copy()
    if len(data) < 10:
        raise ValueError("Fewer than 10 rows remain after excluding missing target values.")
    X = data.drop(columns=[target]).copy()
    y = data[target].copy()
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after excluding the target column.")

    # Convert datetime columns into numeric calendar components.
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            dt = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}__year"] = dt.dt.year
            X[f"{col}__month"] = dt.dt.month
            X[f"{col}__day"] = dt.dt.day
            X.drop(columns=[col], inplace=True)

    # Remove columns that are entirely empty or constant; keep target untouched.
    usable = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    X = X[usable]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns were found (all features are empty or constant).")

    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
        ]), categorical))
    prep = ColumnTransformer(transformers, remainder="drop")

    if model_type == "REGRESSION":
        y_num = pd.to_numeric(y, errors="coerce")
        valid = y_num.notna()
        X, y_num = X.loc[valid], y_num.loc[valid]
        if len(y_num) < 10:
            raise ValueError("The selected target must contain at least 10 valid numeric values for regression.")
        if y_num.nunique() < 2:
            raise ValueError("Regression target must contain at least two different numeric values.")
        X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        results, fitted = [], []
        for name, estimator, grid in _regression_candidates():
            try:
                gs = _search(estimator, grid, prep, X_train, y_train, "r2", cv)
                pipe = gs.best_estimator_
                pred = pipe.predict(X_test)
                rmse = mean_squared_error(y_test, pred) ** 0.5
                metrics = {"r2": _json_number(r2_score(y_test, pred)), "mae": _json_number(mean_absolute_error(y_test, pred)), "rmse": _json_number(rmse)}
                rank = _json_number(gs.best_score_)
                entry = {"model": name, "hyperparameters": _clean_params(gs.best_params_), "threshold": None, **metrics}
                results.append(entry)
                fitted.append((rank if rank is not None else -float("inf"), name, pipe, metrics, entry))
            except Exception as exc:
                continue
        if not fitted:
            raise ValueError("All candidate regression models failed. Check the dataset and feature values.")
        fitted.sort(key=lambda item: item[0], reverse=True)
        _, best_name, best_pipe, best_metrics, _ = fitted[0]
        results = [item[4] for item in fitted[:TOP_N]]
        best_threshold = None
        best_params = fitted[0][4]["hyperparameters"]
        score_name = "R²"
    else:
        # Preserve original class labels while sklearn receives integer labels.
        if y.nunique() < 2:
            raise ValueError("Classification target must contain at least two classes.")
        counts = y.value_counts(dropna=True)
        stratify = y if counts.min() >= 2 else None
        encoder = LabelEncoder()
        y_encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
        test_count = max(1, int(round(len(y_encoded) * 0.2)))
        if stratify is not None and test_count < y_encoded.nunique():
            stratify = None
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=stratify)
        is_binary = y_encoded.nunique() == 2
        if y_train.value_counts().min() >= 3:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
        results, fitted = [], []
        for name, estimator, grid in _classification_candidates():
            try:
                gs = _search(estimator, grid, prep, X_train, y_train, "f1_weighted", cv)
                pipe = gs.best_estimator_
                threshold = None
                rank = _json_number(gs.best_score_)
                if is_binary:
                    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
                    classes = np.asarray(pipe.classes_)
                    best_thr, best_thr_score = 0.5, -1.0
                    for thr in THRESHOLDS:
                        s = f1_score(y_train, classes[(oof >= thr).astype(int)], average="weighted", zero_division=0)
                        if s > best_thr_score:
                            best_thr, best_thr_score = thr, s
                    threshold = best_thr
                    rank = _json_number(best_thr_score)
                    final_model = ThresholdClassifier(pipe, threshold)
                else:
                    final_model = pipe
                pred = final_model.predict(X_test)
                metrics = {
                    "accuracy": _json_number(accuracy_score(y_test, pred)),
                    "precision_weighted": _json_number(precision_score(y_test, pred, average="weighted", zero_division=0)),
                    "recall_weighted": _json_number(recall_score(y_test, pred, average="weighted", zero_division=0)),
                    "f1_weighted": _json_number(f1_score(y_test, pred, average="weighted", zero_division=0)),
                }
                entry = {"model": name, "hyperparameters": _clean_params(gs.best_params_), "threshold": threshold, **metrics}
                fitted.append((rank if rank is not None else -float("inf"), name, final_model, metrics, entry))
            except Exception as exc:
                continue
        if not fitted:
            raise ValueError("All candidate classification models failed. Check the dataset and feature values.")
        fitted.sort(key=lambda item: item[0], reverse=True)
        _, best_name, best_pipe, best_metrics, _ = fitted[0]
        results = [item[4] for item in fitted[:TOP_N]]
        best_threshold = fitted[0][4]["threshold"]
        best_params = fitted[0][4]["hyperparameters"]
        score_name = "Weighted F1"

    model_id = f"model_{uuid.uuid4().hex[:12]}.joblib"
    model_path = MODEL_DIR / model_id
    joblib.dump({"pipeline": best_pipe, "target_column": target, "model_type": model_type, "feature_columns": list(X.columns), "threshold": best_threshold, "hyperparameters": best_params}, model_path)
    return {
        "status": "success", "model_id": model_id, "target_column": target,
        "model_type": model_type, "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]), "best_model": best_name,
        "selection_metric": score_name, "best_metrics": best_metrics,
        "models": results,
        "note": "Scores are measured on a held-out test split. The target column was excluded from preprocessing and features."
    }