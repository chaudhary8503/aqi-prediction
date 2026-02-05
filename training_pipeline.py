"""
training_pipeline.py

Runs the "training pipeline" for the AQI project:
1) Read historical (features, target) from Hopsworks Feature Store
2) Train + evaluate models (Ridge, RandomForest, GradientBoosting, SVR)
3) Store the best model in Hopsworks Model Registry

Designed to run daily via GitHub Actions / cron.

Environment variables (required):
- HOPSWORKS_API_KEY
- HOPSWORKS_PROJECT
- HOPSWORKS_HOST (default: c.app.hopsworks.ai)
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


FEATURE_COLS = [
    "hour", "day", "month",
    "temperature", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "o3",
    "aqi_change_rate",
]
TARGET_COL = "aqi_target"


def require_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def hopsworks_login() -> Any:
    require_env("HOPSWORKS_API_KEY")
    require_env("HOPSWORKS_PROJECT")
    os.environ.setdefault("HOPSWORKS_HOST", os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"))
    return hopsworks.login()


def time_aware_split(df: pd.DataFrame, time_col: str = "event_time", train_frac: float = 0.8):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, format="mixed")
    df = df.sort_values(time_col)
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    project = hopsworks_login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()  # offline read
    if df is None or len(df) < 10:
        raise RuntimeError(f"Not enough data to train. Rows found: {0 if df is None else len(df)}")

    # Clean rows
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    train_df, test_df = time_aware_split(df, time_col="event_time", train_frac=0.8)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42),
        "gbr": GradientBoostingRegressor(),
        "svr": SVR(kernel="rbf"),
    }

    results: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        print(name, metrics)

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None
    print(f"\nBest model: {best_name} RMSE: {best_rmse}")

    # Save artifacts
    model_dir = "aqi_model_artifact"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/model.joblib")
    joblib.dump(FEATURE_COLS, f"{model_dir}/feature_cols.joblib")
    with open(f"{model_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # Register in Hopsworks Model Registry
    mr = project.get_model_registry()
    model = mr.python.create_model(
        name="aqi_predictor",
        metrics=results[best_name],
        description="AQI predictor trained from Hopsworks feature store backfill data",
    )
    model.save(model_dir)
    print("Saved model to registry:", model.name)


if __name__ == "__main__":
    main()
