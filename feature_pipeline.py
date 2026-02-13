"""
feature_pipeline.py

Runs the "feature pipeline" for the AQI project:
1) Fetch raw weather + pollutant data (OpenWeather current endpoints)
2) Compute features + target (AQI)
3) Store row(s) in Hopsworks Feature Store (Feature Group: aqi_features v1)

Designed to run hourly via GitHub Actions / cron.
Optionally supports backfilling historical rows if BACKFILL_DAYS is set.

Environment variables:
- OPENWEATHER_API_KEY (required)
- CITY (default: Karachi)
- COUNTRY_CODE (default: PK)

Hopsworks (required for storage):
- HOPSWORKS_API_KEY
- HOPSWORKS_PROJECT
- HOPSWORKS_HOST (default: c.app.hopsworks.ai)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple, Optional

import requests
import pandas as pd
import hopsworks


# -----------------------------
# External APIs
# -----------------------------
OW_BASE = "https://api.openweathermap.org"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _get(url: str, params: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def geocode_city(city_name: str, api_key: str, country_code: Optional[str] = None) -> Tuple[float, float, str]:
    q = city_name if not country_code else f"{city_name},{country_code}"
    data = _get(f"{OW_BASE}/geo/1.0/direct", {"q": q, "limit": 1, "appid": api_key})
    if not data:
        raise ValueError(f"City not found: {q}")
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("name", city_name)


def fetch_current_weather(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    return _get(f"{OW_BASE}/data/2.5/weather", {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})


def fetch_current_air_pollution(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    return _get(f"{OW_BASE}/data/2.5/air_pollution", {"lat": lat, "lon": lon, "appid": api_key})


def fetch_pollution_history(lat: float, lon: float, start_dt: datetime, end_dt: datetime, api_key: str) -> Dict[str, Any]:
    """OpenWeather air pollution history."""
    return _get(
        f"{OW_BASE}/data/2.5/air_pollution/history",
        {
            "lat": lat,
            "lon": lon,
            "start": int(start_dt.timestamp()),
            "end": int(end_dt.timestamp()),
            "appid": api_key,
        },
    )


def fetch_weather_archive_hourly(lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, Any]:
    """Open-Meteo archive: hourly weather (UTC), no key."""
    return _get(
        OM_ARCHIVE,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "timezone": "UTC",
        },
    )


# -----------------------------
# Feature computation
# -----------------------------
def fetch_raw_city_data(city_name: str, api_key: str, country_code: Optional[str] = None) -> Dict[str, Any]:
    lat, lon, resolved_name = geocode_city(city_name, api_key, country_code=country_code)
    weather = fetch_current_weather(lat, lon, api_key)
    pollution = fetch_current_air_pollution(lat, lon, api_key)
    return {
        "city": resolved_name,
        "lat": lat,
        "lon": lon,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "weather_raw": weather,
        "pollution_raw": pollution,
    }


def compute_features_and_target(raw: Dict[str, Any], previous_aqi: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ts = datetime.fromisoformat(raw["fetched_at_utc"])
    weather = raw["weather_raw"]
    pollution = raw["pollution_raw"]["list"][0]
    comps = pollution["components"]

    aqi = int(pollution["main"]["aqi"])
    aqi_change = 0.0 if previous_aqi is None else float(aqi - previous_aqi)

    features = {
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        "temperature": float(weather["main"]["temp"]),
        "humidity": int(weather["main"]["humidity"]),
        "pressure": int(weather["main"]["pressure"]),
        "wind_speed": float(weather["wind"]["speed"]),
        "pm2_5": float(comps.get("pm2_5")) if comps.get("pm2_5") is not None else None,
        "pm10": float(comps.get("pm10")) if comps.get("pm10") is not None else None,
        "no2": float(comps.get("no2")) if comps.get("no2") is not None else None,
        "o3": float(comps.get("o3")) if comps.get("o3") is not None else None,
        "aqi_change_rate": float(aqi_change),
    }
    target = {"aqi": aqi}
    return features, target


def build_current_row(raw: Dict[str, Any], features: Dict[str, Any], target: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "city": raw["city"],
        "event_time": raw["fetched_at_utc"],  # string primary key
        **features,
        "aqi_target": target["aqi"],
    }
    return pd.DataFrame([row])


def build_backfill_rows(city: str, country: str, days: int, api_key: str) -> pd.DataFrame:
    lat, lon, resolved = geocode_city(city, api_key, country_code=country)

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    pol = fetch_pollution_history(lat, lon, start_dt, end_dt, api_key)
    pol_list = pol.get("list", [])

    w = fetch_weather_archive_hourly(lat, lon, start_dt.date().isoformat(), end_dt.date().isoformat())
    wh = w["hourly"]
    weather_by_time: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(wh["time"]):
        weather_by_time[t + ":00"] = {
            "temperature": wh["temperature_2m"][i],
            "humidity": wh["relative_humidity_2m"][i],
            "pressure": wh["surface_pressure"][i],
            "wind_speed": wh["wind_speed_10m"][i],
        }

    rows = []
    pol_list_sorted = sorted(pol_list, key=lambda x: x["dt"])
    prev_aqi = None

    for item in pol_list_sorted:
        ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        k = ts.strftime("%Y-%m-%dT%H:00:00")
        if k not in weather_by_time:
            continue

        aqi = int(item["main"]["aqi"])
        aqi_change = 0.0 if prev_aqi is None else float(aqi - prev_aqi)
        prev_aqi = aqi
        comps = item["components"]

        rows.append(
            {
                "city": resolved,
                "event_time": ts.isoformat(),
                "hour": ts.hour,
                "day": ts.day,
                "month": ts.month,
                "temperature": weather_by_time[k]["temperature"],
                "humidity": weather_by_time[k]["humidity"],
                "pressure": weather_by_time[k]["pressure"],
                "wind_speed": weather_by_time[k]["wind_speed"],
                "pm2_5": comps.get("pm2_5"),
                "pm10": comps.get("pm10"),
                "no2": comps.get("no2"),
                "o3": comps.get("o3"),
                "aqi_change_rate": aqi_change,
                "aqi_target": aqi,
            }
        )

    return pd.DataFrame(rows)


def cast_to_feature_group_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match your existing Feature Group schema exactly:

    city: string (pk)
    event_time: string (pk)
    hour/day/month: bigint
    temperature: double
    humidity: bigint
    pressure: bigint
    wind_speed: double
    pm2_5/pm10/no2/o3: double
    aqi_change_rate: double
    aqi_target: bigint
    """
    out = df.copy()

    out["city"] = out["city"].astype("string")
    out["event_time"] = out["event_time"].astype("string")

    bigint_cols = ["hour", "day", "month", "humidity", "pressure", "aqi_target"]
    for c in bigint_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64").astype("int64")

    double_cols = ["temperature", "wind_speed", "pm2_5", "pm10", "no2", "o3", "aqi_change_rate"]
    for c in double_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    return out


# -----------------------------
# Hopsworks storage
# -----------------------------
def require_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def hopsworks_login() -> Any:
    # hopsworks.login() will pick up these env vars if present
    # (we still validate so CI fails with a clear message)
    require_env("HOPSWORKS_API_KEY")
    require_env("HOPSWORKS_PROJECT")
    os.environ.setdefault("HOPSWORKS_HOST", os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"))
    return hopsworks.login()


def get_or_create_fg(fs: Any) -> Any:
    return fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "event_time"],
        description="AQI features built from OpenWeather raw data",
        online_enabled=False,
    )


def retry_fg_insert(fg: Any, df: pd.DataFrame, retries: int = 3, base_sleep_s: int = 10) -> Any:
    """
    Hopsworks can occasionally drop the connection when starting the materialization job.
    This retry handles transient network failures like:
    - RemoteDisconnected
    - ConnectionError / ProtocolError
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            return fg.insert(df)
        except requests.exceptions.RequestException as e:
            last_err = e
            sleep_s = base_sleep_s * attempt
            print(f"[WARN] fg.insert failed (attempt {attempt}/{retries}): {e}")
            print(f"[WARN] sleeping {sleep_s}s then retrying...")
            time.sleep(sleep_s)
    # If we reach here, all retries failed
    if last_err is not None:
        raise last_err
    raise RuntimeError("fg.insert failed but no exception was captured.")


def main() -> None:
    ow_key = require_env("OPENWEATHER_API_KEY")
    city = os.getenv("CITY", "Karachi")
    country = os.getenv("COUNTRY_CODE", "PK")

    # 1) Always insert the current row (hourly run)
    raw = fetch_raw_city_data(city, ow_key, country_code=country)
    features, target = compute_features_and_target(raw)
    df_now = cast_to_feature_group_schema(build_current_row(raw, features, target))

    project = hopsworks_login()
    fs = project.get_feature_store()
    fg = get_or_create_fg(fs)

    # Retry insert to avoid random CI failures due to transient Hopsworks disconnects
    retry_fg_insert(fg, df_now)
    print(f"Inserted current row: city={raw['city']} event_time={raw['fetched_at_utc']}")

    # 2) Optional backfill (run manually or in a one-off workflow)
    backfill_days = os.getenv("BACKFILL_DAYS")
    if backfill_days:
        days = int(backfill_days)
        df_hist = build_backfill_rows(city=city, country=country, days=days, api_key=ow_key)
        if len(df_hist) == 0:
            print("Backfill produced 0 rows. Skipping insert.")
            return
        df_hist = cast_to_feature_group_schema(df_hist)

        # Retry backfill insert as well
        retry_fg_insert(fg, df_hist)
        print(f"Inserted backfill rows: {len(df_hist)} (days={days})")


if __name__ == "__main__":
    main()
