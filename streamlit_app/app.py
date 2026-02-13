# app.py
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import hopsworks


FG_NAME = "aqi_features"
FG_VERSION = 1
MODEL_NAME = "aqi_predictor"

MODEL_FILE = "model.joblib"
COLS_FILE = "feature_cols.joblib"


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return str(v)


@st.cache_resource(show_spinner=False)
def get_project():
    return hopsworks.login(
        host=_env("HOPSWORKS_HOST"),
        project=_env("HOPSWORKS_PROJECT"),
        api_key_value=_env("HOPSWORKS_API_KEY"),
    )


@st.cache_resource(show_spinner=False)
def load_latest_model_and_cols():
    project = get_project()
    mr = project.get_model_registry()

    models = mr.get_models(MODEL_NAME)
    if not models:
        raise RuntimeError(f"No models found in registry with name='{MODEL_NAME}'")

    latest = max(models, key=lambda m: int(m.version))
    model_dir = Path(latest.download())

    model_path = model_dir / MODEL_FILE
    cols_path = model_dir / COLS_FILE

    if not model_path.exists():
        raise RuntimeError(f"Missing artifact: {model_path}")
    if not cols_path.exists():
        raise RuntimeError(f"Missing artifact: {cols_path}")

    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)

    return model, feature_cols, int(latest.version)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_city_rows(city: str) -> tuple[pd.Series, pd.DataFrame]:
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)

    # For ~5k rows this is fine. If this grows, switch to Feature View + online store.
    df = fg.read()
    if df.empty:
        raise RuntimeError("Feature group is empty.")

    df = df[df["city"].astype(str) == str(city)].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for city='{city}'")

    # Your format: 2026-02-03T17:00:00+00:00
    df["event_time_ts"] = pd.to_datetime(df["event_time"], format="ISO8601", utc=True)
    df = df.sort_values("event_time_ts")

    latest_row = df.iloc[-1]
    recent_df = df.tail(72)  # last ~3 days (72 hours)
    return latest_row, recent_df


def aqi_category_openweather(aqi_1_to_5: float) -> str:
    # OpenWeather AQI index mapping: 1..5
    try:
        v = int(round(float(aqi_1_to_5)))
    except Exception:
        return "Unknown"

    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }.get(v, "Unknown")


def clear_caches_and_rerun():
    fetch_city_rows.clear()
    load_latest_model_and_cols.clear()
    get_project.clear()
    st.rerun()


st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")
st.title("🌫️ AQI Prediction Dashboard")

with st.sidebar:
    st.header("Settings")
    city = st.text_input("City", value=os.getenv("CITY", "Karachi"))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Refresh"):
            clear_caches_and_rerun()
    with col_b:
        st.caption("Caches: model (resource), data (5 min)")

try:
    model, feature_cols, model_version = load_latest_model_and_cols()
    latest_row, recent_df = fetch_city_rows(city)

    x = pd.DataFrame([latest_row]).copy()
    missing = [c for c in feature_cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns in latest row: {missing}")

    y_pred = float(model.predict(x[feature_cols])[0])
    category = aqi_category_openweather(y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted AQI", f"{y_pred:.2f}")
    m2.metric("Category", category)
    m3.metric("Model Version", str(model_version))

    st.subheader("Latest Feature Snapshot")

    left, right = st.columns([1, 1])

    with left:
        st.write("**City**:", str(latest_row.get("city")))
        st.write("**Event Time (UTC)**:", str(latest_row.get("event_time")))
        key_cols = [
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "pm2_5",
            "pm10",
            "no2",
            "o3",
            "aqi_change_rate",
        ]
        snap = {k: latest_row.get(k) for k in key_cols if k in latest_row.index}
        st.dataframe(pd.DataFrame([snap]), use_container_width=True)

    with right:
        st.subheader("Last ~3 days (model predictions)")
        recent = recent_df.copy()
        recent["event_time_ts"] = pd.to_datetime(recent["event_time"], format="ISO8601", utc=True)
        recent = recent.sort_values("event_time_ts")

        # ensure completeness for prediction
        recent = recent[recent[feature_cols].notna().all(axis=1)].copy()

        if recent.empty:
            st.info("Not enough complete rows to plot a trend.")
        else:
            recent["pred_aqi"] = model.predict(recent[feature_cols]).astype(float)
            chart_df = recent[["event_time_ts", "pred_aqi"]].set_index("event_time_ts")
            st.line_chart(chart_df, use_container_width=True)

except Exception as e:
    st.error(str(e))
    st.stop()
