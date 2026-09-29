"""Central configuration: paths, model settings and business thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Kaggle datasets (downloaded once with kagglehub, then copied into data/raw)
KAGGLE_POWER = "anikannal/solar-power-generation-data"
KAGGLE_IMAGES = "pythonafroz/solar-panel-images"

# ---- Numerical model --------------------------------------------------------
POWER_MODEL_PATH = MODELS_DIR / "power_rf.joblib"
POWER_METRICS_PATH = RESULTS_DIR / "power_metrics.json"
FEATURES = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR_SIN",
    "HOUR_COS",
    "PLANT",
]
TARGET = "AC_POWER"

# Performance ratio (actual / expected energy) thresholds per inverter per day
PR_WARNING = 0.90
PR_CRITICAL = 0.75

# ---- Image model -------------------------------------------------------------
IMAGE_MODEL_PATH = MODELS_DIR / "panel_resnet.pt"
IMAGE_METRICS_PATH = RESULTS_DIR / "image_metrics.json"
IMAGE_SIZE = 288
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered",
]

# ---- Chatbot (Ollama) ---------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma2:9b"          # falls back to OLLAMA_FALLBACK if not pulled
OLLAMA_FALLBACK = "gemma2:2b"

# ---- Live weather (Open-Meteo, free, no key) ----------------------------------
DEFAULT_LAT = 31.95   # Amman
DEFAULT_LON = 35.91
PANEL_NOCT = 45.0     # nominal operating cell temperature, used to estimate module temp

# ---- API (used by the dashboard to reach the backend) ------------------------
API_URL = "http://127.0.0.1:8000"
