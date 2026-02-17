import os
from pathlib import Path
from pyexpat import model

import joblib
import pandas as pd
import shap
import hopsworks
import matplotlib.pyplot as plt


FG_NAME = "aqi_features"
FG_VERSION = 1
MODEL_NAME = "aqi_predictor"

MODEL_FILE = "model.joblib"
COLS_FILE = "feature_cols.joblib"


def get_project():
    host = os.getenv("HOPSWORKS_HOST").replace("https://", "").replace("http://", "").rstrip("/")
    return hopsworks.login(
        host=host,
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )


def load_latest_model(project):
    mr = project.get_model_registry()
    models = mr.get_models(MODEL_NAME)

    if not models:
        raise RuntimeError("No model found in registry.")

    latest = max(models, key=lambda m: int(m.version))
    model_dir = Path(latest.download())

    model = joblib.load(model_dir / MODEL_FILE)
    feature_cols = joblib.load(model_dir / COLS_FILE)

    return model, feature_cols


def load_recent_data(project, feature_cols, city):
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)

    df = fg.read()
    df = df[df["city"] == city].copy()

    df["event_time_ts"] = pd.to_datetime(df["event_time"], format="ISO8601", utc=True)
    df = df.sort_values("event_time_ts")

    # Use last 200 rows for SHAP (enough, but not too heavy)
    df = df.tail(200)

    df = df[df[feature_cols].notna().all(axis=1)]

    return df[feature_cols]


def main():
    print("Connecting to Hopsworks...")
    project = get_project()

    print("Loading latest model...")
    model, feature_cols = load_latest_model(project)

    city = os.getenv("CITY", "Karachi")
    print(f"Loading recent data for city: {city}")

    X = load_recent_data(project, feature_cols, city)

    if X.empty:
        raise RuntimeError("No usable data found for SHAP.")

    print("Running SHAP...")

    # TreeExplainer works for RandomForest / GradientBoosting
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    

    print("Generating SHAP summary plot...")

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)

    print("Saved: shap_summary.png")



if __name__ == "__main__":
    main()
