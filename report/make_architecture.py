"""Draw the system architecture diagram for the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "architecture.png"

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=170)
ax.set_xlim(0, 100)
ax.set_ylim(0, 59)
ax.axis("off")

INPUT, API, MODEL, LOGIC, UI, USER = "#fff3c4", "#d6e8ff", "#d7f5dd", "#ffe2cc", "#e9dcff", "#eeeeee"


def box(x, y, w, h, text, color, bold=False, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec="#555555", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x1, y1, x2, y2, label="", both=False, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>",
                                 mutation_scale=10, color="#444444", lw=1))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7, color="#333333",
                ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


# inputs
ax.text(9, 56.5, "INPUTS", ha="center", fontsize=8, fontweight="bold", color="#777777")
box(1, 43, 16, 7, "Inverter power +\nweather sensors\n(15-min readings)", INPUT)
box(1, 31, 16, 7, "Panel photo\n(drone / technician)", INPUT)
box(1, 19, 16, 7, "Open-Meteo\nlive forecast", INPUT)
box(1, 7, 16, 7, "Operator question\n(Arabic / English)", INPUT)

# API
box(29, 7, 13, 43, "FastAPI\nbackend\n\n/predict/power\n/fleet/*\n/analyze/image\n/diagnose\n/forecast\n/chat", API,
    bold=False, size=8)
ax.text(35.5, 56.5, "API LAYER", ha="center", fontsize=8, fontweight="bold", color="#777777")

# models
ax.text(66, 56.5, "AI COMPONENTS", ha="center", fontsize=8, fontweight="bold", color="#777777")
box(52.5, 43, 27, 7, "Random Forest\nexpected power → PR, kWh lost", MODEL, bold=True, size=8)
box(52.5, 31, 27, 7, "ResNet-18 (CNN)\npanel condition, 6 classes", MODEL, bold=True, size=8)
box(52.5, 19, 27, 7, "Decision engine\naction · team · priority · cost", LOGIC, bold=True, size=8)
box(52.5, 7, 27, 7, "Gemma 2 Transformer\n(Ollama) chatbot", UI, bold=True, size=8)

# UI + user
box(86, 25, 12, 12, "Streamlit\ndashboard", API, bold=True)
box(86, 7, 12, 8, "Operator /\nmaintenance\nteam", USER)

# input -> API
for y in (46.5, 34.5, 22.5, 10.5):
    arrow(18, y, 28.4, y)
# API <-> models
arrow(42.6, 46.5, 51.9, 46.5, "how much?", both=True, ly=1.6)
arrow(42.6, 34.5, 51.9, 34.5, "why?", both=True, ly=1.6)
arrow(42.6, 22.5, 51.9, 22.5, "what to do?", both=True, ly=1.6)
arrow(42.6, 10.5, 51.9, 10.5, "live context", both=True, ly=1.6)
# models feeding the decision engine
arrow(66, 30.4, 66, 26.6)
ax.add_patch(FancyArrowPatch((80.1, 46.5), (82, 46.5), arrowstyle="-", color="#444444", lw=1))
ax.add_patch(FancyArrowPatch((82, 46.5), (82, 24), arrowstyle="-", color="#444444", lw=1))
ax.add_patch(FancyArrowPatch((82, 24), (80.1, 24), arrowstyle="-|>", mutation_scale=10, color="#444444", lw=1))
# API -> UI -> user
ax.add_patch(FancyArrowPatch((35.5, 51), (35.5, 53.5), arrowstyle="-", color="#1f5fa8", lw=1.2, ls="--"))
ax.add_patch(FancyArrowPatch((35.5, 53.5), (92, 53.5), arrowstyle="-", color="#1f5fa8", lw=1.2, ls="--"))
ax.add_patch(FancyArrowPatch((92, 53.5), (92, 37.6), arrowstyle="<|-|>", mutation_scale=10, color="#1f5fa8", lw=1.2))
ax.text(64, 54.4, "HTTP / JSON", fontsize=7, color="#1f5fa8", style="italic")
arrow(92, 24.4, 92, 15.6, both=True)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
