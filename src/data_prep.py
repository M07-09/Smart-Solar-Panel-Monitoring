"""Load, clean and merge the Kaggle solar power generation + weather sensor data.

Output: data/processed/solar_merged.csv — one row per inverter per 15-min interval,
with weather readings, time features and a `HEALTHY` flag used for training.
"""
import shutil

import numpy as np
import pandas as pd

from config import KAGGLE_POWER, PROCESSED_DIR, RAW_DIR

FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]


def ensure_raw_data():
    """Download the dataset with kagglehub (cached) and copy the CSVs into data/raw."""
    if all((RAW_DIR / f).exists() for f in FILES):
        return
    import kagglehub

    src = kagglehub.dataset_download(KAGGLE_POWER)
    for f in FILES:
        shutil.copy(f"{src}/{f}", RAW_DIR / f)


def load_plant(n: int) -> pd.DataFrame:
    gen = pd.read_csv(RAW_DIR / f"Plant_{n}_Generation_Data.csv")
    wea = pd.read_csv(RAW_DIR / f"Plant_{n}_Weather_Sensor_Data.csv")
    # Plant 1 generation uses dd-mm-yyyy HH:MM, everything else is ISO
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=(n == 1), format="mixed")
    wea["DATE_TIME"] = pd.to_datetime(wea["DATE_TIME"])

    wea = wea[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df = gen.merge(wea, on="DATE_TIME", how="inner")
    df["PLANT"] = n
    df = df.rename(columns={"SOURCE_KEY": "INVERTER"})
    return df[["DATE_TIME", "PLANT", "INVERTER", "AC_POWER", "DC_POWER",
               "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60
    df["HOUR"] = hour
    df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)
    df["DATE"] = df["DATE_TIME"].dt.date
    return df


def flag_healthy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Mark rows that represent normal operation, so the model learns *expected* output.

    Excluded from training:
      * daylight outages (irradiation clearly > 0 but AC power == 0)
      * temporary faults: an inverter producing < 60 % of its plant's median inverter
        at the same timestamp (while the plant is producing)
      * inverters whose long-run output is < 85 % of their plant's median inverter
        (chronic underperformers — exactly what the system must later detect)
    """
    day = df["IRRADIATION"] > 0.05
    ts_median = df.groupby(["PLANT", "DATE_TIME"])["AC_POWER"].transform("median")
    outage = day & ((df["AC_POWER"] <= 0) | ((ts_median > 50) & (df["AC_POWER"] < 0.6 * ts_median)))

    daylight = df[day & ~outage]
    inv_mean = daylight.groupby(["PLANT", "INVERTER"])["AC_POWER"].mean()
    ratio = inv_mean / inv_mean.groupby("PLANT").transform("median")
    weak = set(ratio[ratio < 0.85].index)

    df["HEALTHY"] = ~outage & ~df.set_index(["PLANT", "INVERTER"]).index.isin(weak)
    return df, ratio.sort_values()


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    ensure_raw_data()
    df = pd.concat([load_plant(1), load_plant(2)], ignore_index=True)
    df = df.dropna().drop_duplicates(subset=["DATE_TIME", "PLANT", "INVERTER"])
    df = df[(df["AC_POWER"] >= 0) & (df["IRRADIATION"] >= 0)]
    df = add_time_features(df)
    df, ratio = flag_healthy(df)
    df = df.sort_values(["DATE_TIME", "PLANT", "INVERTER"]).reset_index(drop=True)
    df.to_csv(PROCESSED_DIR / "solar_merged.csv", index=False)
    return df, ratio


if __name__ == "__main__":
    df, ratio = build_dataset()
    print("rows:", len(df), "| healthy rows:", int(df["HEALTHY"].sum()))
    print("period:", df["DATE_TIME"].min(), "->", df["DATE_TIME"].max())
    print("inverter output vs plant median (lowest 6):")
    print(ratio.head(6).round(3).to_string())
