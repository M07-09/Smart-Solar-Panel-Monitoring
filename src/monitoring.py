"""Fleet monitoring built on the numerical model.

For every inverter and every day it compares ACTUAL energy with the EXPECTED energy
predicted by the Random Forest, giving a Performance Ratio (PR = actual / expected).
PR below the thresholds in config marks the inverter as Warning / Critical.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import FEATURES, PR_CRITICAL, PR_WARNING, PROCESSED_DIR, POWER_MODEL_PATH, RESULTS_DIR

DAILY_PR_PATH = RESULTS_DIR / "inverter_daily_pr.csv"
INTERVAL_H = 0.25  # readings are every 15 minutes


@lru_cache(maxsize=1)
def power_model():
    return joblib.load(POWER_MODEL_PATH)


def predict_power(frame: pd.DataFrame) -> np.ndarray:
    """Expected AC power (kW) for rows holding the model FEATURES."""
    return power_model().predict(frame[FEATURES]).clip(min=0)


def status_from_pr(pr: float) -> str:
    if pr < PR_CRITICAL:
        return "Critical"
    if pr < PR_WARNING:
        return "Warning"
    return "Normal"


@lru_cache(maxsize=1)
def readings() -> pd.DataFrame:
    """All 15-min readings with the model's expected power attached."""
    df = pd.read_csv(PROCESSED_DIR / "solar_merged.csv", parse_dates=["DATE_TIME"])
    df["EXPECTED"] = predict_power(df)
    # Friendly inverter names: P1-INV01 ... P2-INV22
    names = {}
    for plant, grp in df.groupby("PLANT"):
        for i, key in enumerate(sorted(grp["INVERTER"].unique()), 1):
            names[key] = f"P{plant}-INV{i:02d}"
    df["NAME"] = df["INVERTER"].map(names)
    df["DATE"] = df["DATE_TIME"].dt.date.astype(str)
    return df


@lru_cache(maxsize=1)
def daily_pr() -> pd.DataFrame:
    """Daily actual vs expected energy (kWh) and PR per inverter."""
    df = readings()
    day = df[df["IRRADIATION"] > 0.05]
    g = day.groupby(["DATE", "PLANT", "NAME"]).agg(
        ACTUAL_KWH=("AC_POWER", "sum"), EXPECTED_KWH=("EXPECTED", "sum"),
        IRRADIATION=("IRRADIATION", "mean"),
    ).reset_index()
    g[["ACTUAL_KWH", "EXPECTED_KWH"]] *= INTERVAL_H
    g["PR"] = (g["ACTUAL_KWH"] / g["EXPECTED_KWH"].replace(0, np.nan)).fillna(1.0).clip(0, 1.5)
    g["LOSS_KWH"] = (g["EXPECTED_KWH"] - g["ACTUAL_KWH"]).clip(lower=0)
    g["STATUS"] = g["PR"].map(status_from_pr)
    g.to_csv(DAILY_PR_PATH, index=False)
    return g


def available_dates() -> list[str]:
    return sorted(daily_pr()["DATE"].unique())


def fleet_on(date: str) -> pd.DataFrame:
    return daily_pr().query("DATE == @date").sort_values("PR").reset_index(drop=True)


def inverter_day(name: str, date: str) -> pd.DataFrame:
    df = readings()
    cols = ["DATE_TIME", "AC_POWER", "EXPECTED", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
    return df[(df["NAME"] == name) & (df["DATE"] == date)][cols]


def inverter_history(name: str) -> pd.DataFrame:
    return daily_pr().query("NAME == @name").sort_values("DATE")


def plant_day(date: str) -> pd.DataFrame:
    """Plant-level actual vs expected power curve for one day."""
    df = readings()
    d = df[df["DATE"] == date]
    return d.groupby(["DATE_TIME", "PLANT"])[["AC_POWER", "EXPECTED", "IRRADIATION"]].sum().reset_index()


def fleet_summary(date: str) -> dict:
    f = fleet_on(date)
    return {
        "date": date,
        "inverters": int(len(f)),
        "normal": int((f["STATUS"] == "Normal").sum()),
        "warning": int((f["STATUS"] == "Warning").sum()),
        "critical": int((f["STATUS"] == "Critical").sum()),
        "actual_kwh": round(float(f["ACTUAL_KWH"].sum()), 1),
        "expected_kwh": round(float(f["EXPECTED_KWH"].sum()), 1),
        "loss_kwh": round(float(f["LOSS_KWH"].sum()), 1),
        "fleet_pr": round(float(f["ACTUAL_KWH"].sum() / max(f["EXPECTED_KWH"].sum(), 1e-9)), 3),
    }
