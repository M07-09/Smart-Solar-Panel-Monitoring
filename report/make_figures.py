"""Generate all report figures from the saved training results (no retraining)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import monitoring  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, AMBER, RED, GRAY = "#1f5fa8", "#e8871e", "#2e9e5b", "#f0a202", "#d7263d", "#8a8f98"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10})

power = json.loads((RES / "power_metrics.json").read_text())
lstm = json.loads((RES / "lstm_comparison.json").read_text())
image = json.loads((RES / "image_metrics.json").read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# 1. Actual vs predicted (Random Forest, test days)
pred = pd.read_csv(RES / "power_test_predictions.csv")
day = pred[pred["IRRADIATION"] > 0.05].sample(4000, random_state=0)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.scatter(day["AC_POWER"], day["PREDICTED"], s=4, alpha=0.25, color=BLUE)
m = max(day["AC_POWER"].max(), day["PREDICTED"].max())
ax.plot([0, m], [0, m], "--", color=GRAY, lw=1, label="perfect prediction")
ax.set(xlabel="Actual AC power (kW)", ylabel="Predicted AC power (kW)",
       title=f"Random Forest: actual vs predicted (R² = {power['scores']['RandomForest']['R2']:.3f})")
ax.legend(loc="upper left", frameon=False)
save(fig, "rf_actual_vs_pred.png")

# 2. Feature importance
imp = pd.Series(power["feature_importance"]).sort_values()
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.barh(imp.index, imp.values * 100, color=BLUE)
for y, v in enumerate(imp.values * 100):
    ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8)
ax.set(xlabel="Importance (%)", title="Random Forest feature importance", xlim=(0, 110))
save(fig, "rf_feature_importance.png")

# 3. Model comparison (same test rows)
names = ["Linear Regression", "LSTM", "Random Forest"]
mae = [lstm[k]["MAE"] for k in ("LinearRegression", "LSTM", "RandomForest")]   # identical test rows
r2 = [lstm[k]["R2"] for k in ("LinearRegression", "LSTM", "RandomForest")]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
for ax, vals, title, fmt in ((axes[0], mae, "MAE (kW) — lower is better", "{:.1f}"),
                             (axes[1], r2, "R² — higher is better", "{:.3f}")):
    bars = ax.bar(names, vals, color=[GRAY, ORANGE, BLUE])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
axes[1].set_ylim(0.9, 1.0)
save(fig, "numeric_model_comparison.png")

# 4. Fleet performance ratio on the demo day
f = monitoring.fleet_on("2020-05-20").sort_values("PR")
colors = f["STATUS"].map({"Normal": GREEN, "Warning": AMBER, "Critical": RED})
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.bar(f["NAME"], f["PR"] * 100, color=colors)
ax.axhline(90, ls="--", color=AMBER, lw=1)
ax.axhline(75, ls="--", color=RED, lw=1)
ax.set(ylabel="Performance ratio (%)", title="Performance ratio of all 44 inverters on 2020-05-20")
ax.tick_params(axis="x", rotation=90, labelsize=6)
save(fig, "fleet_pr_2020-05-20.png")

# 5. Worst inverter day curve
d = monitoring.inverter_day("P2-INV22", "2020-05-20")
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.plot(d["DATE_TIME"], d["EXPECTED"], color=BLUE, label="Expected (Random Forest)")
ax.fill_between(d["DATE_TIME"], d["AC_POWER"], color=ORANGE, alpha=0.6, label="Actual")
ax.set(ylabel="AC power (kW)", title="P2-INV22 on 2020-05-20: the inverter drops to zero at midday")
ax.legend(frameon=False, loc="upper left")
fig.autofmt_xdate()
save(fig, "worst_inverter_curve.png")

# 6. Image dataset class distribution
cnt = pd.Series(image["images_per_class"])
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.bar(cnt.index, cnt.values, color=BLUE)
for i, v in enumerate(cnt.values):
    ax.text(i, v + 3, str(v), ha="center", fontsize=8)
ax.set(title="Images per class (885 in total)", ylabel="Images")
ax.tick_params(axis="x", rotation=30)
save(fig, "image_class_distribution.png")

# 7. Confusion matrix
cm = np.array(image["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues")
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(len(cm)), image["classes"], rotation=40, ha="right")
ax.set_yticks(range(len(cm)), image["classes"])
ax.set(xlabel="Predicted", ylabel="True", title=f"ResNet-18 confusion matrix (accuracy {image['test_accuracy'] * 100:.1f}%)")
ax.spines[:].set_visible(False)
save(fig, "resnet_confusion_matrix.png")

# 8. Training curves
h = pd.DataFrame(image["history"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
axes[0].plot(h["epoch"], h["train_loss"], label="train", color=BLUE)
axes[0].plot(h["epoch"], h["val_loss"], label="validation", color=ORANGE)
axes[1].plot(h["epoch"], h["train_acc"] * 100, label="train", color=BLUE)
axes[1].plot(h["epoch"], h["val_acc"] * 100, label="validation", color=ORANGE)
head_end = h[h["phase"] == "head"]["epoch"].max() + 0.5
for ax, t in zip(axes, ("Loss", "Accuracy (%)")):
    ax.axvline(head_end, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="Epoch", title=t)
    ax.legend(frameon=False)
axes[0].text(head_end + 0.3, axes[0].get_ylim()[1] * 0.92, "fine-tuning starts", fontsize=7, color=GRAY)
save(fig, "resnet_training_curves.png")

# 9. Per-class F1
rep = image["report"]
f1 = pd.Series({c: rep[c]["f1-score"] for c in image["classes"]}).sort_values()
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.barh(f1.index, f1.values * 100, color=BLUE)
for y, v in enumerate(f1.values * 100):
    ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=8)
ax.set(xlabel="F1-score (%)", title="ResNet-18 F1-score per class", xlim=(0, 110))
save(fig, "resnet_f1_per_class.png")
