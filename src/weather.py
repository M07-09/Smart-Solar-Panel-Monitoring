"""Live weather from Open-Meteo (free, no API key) -> tomorrow's expected production.

Open-Meteo gives hourly global irradiance (W/m2) and air temperature. We convert them
to the dataset's units (irradiation in kW/m2), estimate module temperature with the
standard NOCT formula and feed them to the Random Forest.
"""
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from config import FEATURES, PANEL_NOCT, RESULTS_DIR
from monitoring import predict_power

URL = "https://api.open-meteo.com/v1/forecast"
CACHE = RESULTS_DIR / "weather_cache.json"  # last good forecast per location, used when offline


def _frame(h: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "DATE_TIME": pd.to_datetime(h["time"]),
        "GHI_W_M2": h["shortwave_radiation"],
        "AMBIENT_TEMPERATURE": h["temperature_2m"],
        "CLOUD_COVER": h["cloud_cover"],
    })


def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def fetch_forecast(lat: float, lon: float, days: int = 2) -> tuple[pd.DataFrame, str]:
    """Hourly forecast and its source: "live", or "saved <time>" if Open-Meteo is unreachable."""
    key = f"{lat:.2f},{lon:.2f}"
    params = {
        "latitude": lat, "longitude": lon, "forecast_days": days, "timezone": "auto",
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    }
    error = None
    for attempt in range(3):  # slow or unstable connections often succeed on a retry
        try:
            r = requests.get(URL, params=params, timeout=(10, 30))
            r.raise_for_status()
            hourly = r.json()["hourly"]
            cache = _cache()
            cache[key] = {"saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "hourly": hourly}
            CACHE.write_text(json.dumps(cache))
            return _frame(hourly), "live"
        except (requests.RequestException, KeyError, ValueError) as e:
            error = e
            time.sleep(1.5 * (attempt + 1))
    saved = _cache().get(key)
    if saved is None:
        raise RuntimeError(f"Open-Meteo unreachable and no saved forecast for {key}: {error}")
    return _frame(saved["hourly"]), f"saved {saved['saved_at']}"


def expected_production(lat: float, lon: float, plant: int = 1) -> tuple[pd.DataFrame, str]:
    """Hourly expected AC power of ONE healthy inverter for today + tomorrow, plus the data source."""
    df, source = fetch_forecast(lat, lon)
    g = df["GHI_W_M2"].clip(lower=0)
    df["IRRADIATION"] = g / 1000.0
    df["MODULE_TEMPERATURE"] = df["AMBIENT_TEMPERATURE"] + (PANEL_NOCT - 20) / 800 * g
    hour = df["DATE_TIME"].dt.hour
    df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)
    df["PLANT"] = plant
    df["EXPECTED_KW"] = predict_power(df[FEATURES])
    df.loc[df["IRRADIATION"] <= 0, "EXPECTED_KW"] = 0.0
    return df, source


def daily_totals(df: pd.DataFrame) -> list[dict]:
    d = df.assign(DATE=df["DATE_TIME"].dt.date.astype(str)).groupby("DATE").agg(
        EXPECTED_KWH=("EXPECTED_KW", "sum"),          # hourly data -> kW * 1 h
        PEAK_KW=("EXPECTED_KW", "max"),
        AVG_CLOUD=("CLOUD_COVER", "mean"),
        MAX_TEMP=("AMBIENT_TEMPERATURE", "max"),
    ).round(1).reset_index()
    return d.to_dict(orient="records")
