"""Comparison model: LSTM that sees the last 2 hours (8 x 15-min steps) of weather
instead of a single reading. Trained on the same healthy rows and the same
chronological split as the Random Forest; both are scored on the SAME test rows.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import FEATURES, POWER_MODEL_PATH, PROCESSED_DIR, RESULTS_DIR, TARGET

STEPS = 8
SEQ_FEATS = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "HOUR_SIN", "HOUR_COS"]
torch.manual_seed(42)


class PowerLSTM(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + 1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, plant):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], plant], dim=1)).squeeze(1)


def windows(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the previous STEPS weather readings of the plant to every row."""
    out = []
    for plant, g in df.groupby("PLANT"):
        w = g.drop_duplicates("DATE_TIME").set_index("DATE_TIME")[SEQ_FEATS].sort_index()
        w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="15min"))
        lagged = {f"{c}_{k}": w[c].shift(STEPS - 1 - k) for k in range(STEPS) for c in SEQ_FEATS}
        lag = pd.DataFrame(lagged, index=w.index).dropna()
        out.append(g.merge(lag, left_on="DATE_TIME", right_index=True, how="inner"))
    return pd.concat(out, ignore_index=True)


def to_tensors(d: pd.DataFrame, mu, sd, device):
    cols = [f"{c}_{k}" for k in range(STEPS) for c in SEQ_FEATS]
    x = (d[cols].to_numpy(np.float32).reshape(len(d), STEPS, len(SEQ_FEATS)) - mu) / sd
    p = (d["PLANT"].to_numpy(np.float32) - 1.5)[:, None]
    return torch.tensor(x, device=device), torch.tensor(p, device=device)


def score(y, p):
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def main():
    df = pd.read_csv(PROCESSED_DIR / "solar_merged.csv", parse_dates=["DATE_TIME"])
    healthy = df[df["HEALTHY"]]
    df = windows(healthy)
    split = df["DATE_TIME"].max().normalize() - pd.Timedelta(days=6)
    tr, te = df[df["DATE_TIME"] < split], df[df["DATE_TIME"] >= split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = tr[SEQ_FEATS].to_numpy(np.float32)
    mu, sd = base.mean(0), base.std(0) + 1e-6
    scale = float(tr[TARGET].max())
    x_tr, p_tr = to_tensors(tr, mu, sd, device)
    y_tr = torch.tensor(tr[TARGET].to_numpy(np.float32) / scale, device=device)
    x_te, p_te = to_tensors(te, mu, sd, device)

    model = PowerLSTM(len(SEQ_FEATS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    t0, n, bs = time.time(), len(y_tr), 512
    for ep in range(1, 21):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx], p_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0:
            print(f"epoch {ep:2d}  train MSE {total / n:.5f}")
    train_sec = time.time() - t0

    model.eval()
    with torch.no_grad():
        lstm_pred = (model(x_te, p_te).clamp(min=0) * scale).cpu().numpy()
    rf_pred = joblib.load(POWER_MODEL_PATH).predict(te[FEATURES])
    lr_train = healthy[healthy["DATE_TIME"] < split]  # same training rows as train_numeric.py
    lr_pred = LinearRegression().fit(lr_train[FEATURES], lr_train[TARGET]).predict(te[FEATURES]).clip(min=0)
    y = te[TARGET].to_numpy()
    day = te["IRRADIATION"].to_numpy() > 0.05
    res = {
        "test_rows": int(len(te)),
        "lstm_train_seconds": round(train_sec, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "LSTM": score(y, lstm_pred), "RandomForest": score(y, rf_pred), "LinearRegression": score(y, lr_pred),
        "LSTM_daylight": score(y[day], lstm_pred[day]), "RandomForest_daylight": score(y[day], rf_pred[day]),
        "LinearRegression_daylight": score(y[day], lr_pred[day]),
    }
    (RESULTS_DIR / "lstm_comparison.json").write_text(json.dumps(res, indent=2))
    for k in ("LSTM", "RandomForest", "LinearRegression", "LSTM_daylight", "RandomForest_daylight"):
        m = res[k]
        print(f"{k:24s} MAE={m['MAE']:7.2f} RMSE={m['RMSE']:7.2f} R2={m['R2']:.4f}")
    print(f"LSTM trained in {train_sec:.0f}s on {res['device']}")


if __name__ == "__main__":
    main()
