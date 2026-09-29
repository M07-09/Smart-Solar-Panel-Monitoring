"""Train the numerical model: Random Forest that predicts the EXPECTED AC power of an
inverter from weather conditions and time of day.

Linear Regression is trained on the same split as a baseline for comparison.
Split is chronological (last 7 days = test) to mimic real deployment.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import FEATURES, POWER_METRICS_PATH, POWER_MODEL_PATH, PROCESSED_DIR, RESULTS_DIR, TARGET


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(PROCESSED_DIR / "solar_merged.csv", parse_dates=["DATE_TIME"])
    healthy = df[df["HEALTHY"]]

    split = healthy["DATE_TIME"].max().normalize() - pd.Timedelta(days=6)
    train, test = healthy[healthy["DATE_TIME"] < split], healthy[healthy["DATE_TIME"] >= split]
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    print(f"train {len(train):,} rows | test {len(test):,} rows (from {split.date()})")

    results = {}
    lr = LinearRegression().fit(X_tr, y_tr)
    results["LinearRegression"] = metrics(y_te, lr.predict(X_te).clip(min=0))

    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=10,
                               n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    train_sec = time.time() - t0
    rf_pred = rf.predict(X_te)
    results["RandomForest"] = metrics(y_te, rf_pred)

    # Daylight-only scores are the honest ones (night rows are trivially 0)
    day = X_te["IRRADIATION"] > 0.05
    results["RandomForest_daylight"] = metrics(y_te[day], rf_pred[day])
    results["LinearRegression_daylight"] = metrics(y_te[day], lr.predict(X_te[day]).clip(min=0))

    importance = dict(sorted(zip(FEATURES, rf.feature_importances_.round(4).tolist()),
                             key=lambda kv: -kv[1]))
    out = {
        "model": "RandomForestRegressor",
        "params": rf.get_params(),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_from": str(split.date()),
        "train_seconds": round(train_sec, 1),
        "scores": results,
        "feature_importance": importance,
    }
    POWER_METRICS_PATH.write_text(json.dumps(out, indent=2, default=str))
    joblib.dump(rf, POWER_MODEL_PATH, compress=3)

    # Keep a sample of test predictions for the report / dashboard plots
    sample = test.assign(PREDICTED=rf_pred)[["DATE_TIME", "PLANT", "INVERTER", TARGET, "PREDICTED",
                                             "IRRADIATION"]]
    sample.to_csv(RESULTS_DIR / "power_test_predictions.csv", index=False)

    for name, m in results.items():
        print(f"{name:28s} MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:8.2f}  R2={m['R2']:.4f}")
    print("feature importance:", importance)
    print(f"saved -> {POWER_MODEL_PATH} ({POWER_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
