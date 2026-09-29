"""FastAPI backend — the single gateway that connects every AI component.

Streamlit never touches a model directly: it calls these endpoints, and the API
routes data to the Random Forest, the ResNet, the decision engine and Gemma.

Run:  uvicorn api.main:app --port 8000      (from the project root)
Docs: http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402
import math  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import chatbot  # noqa: E402
import decision  # noqa: E402
import monitoring  # noqa: E402
import vision  # noqa: E402
import weather  # noqa: E402
from config import (DEFAULT_LAT, DEFAULT_LON, FEATURES, IMAGE_METRICS_PATH,  # noqa: E402
                    POWER_METRICS_PATH, PR_CRITICAL, PR_WARNING, RESULTS_DIR, ROOT)

LAST_DIAGNOSIS: dict = {}  # latest maintenance ticket produced by /diagnose

app = FastAPI(
    title="Smart Solar Panel Monitoring & Maintenance API",
    description="Random Forest (power) + ResNet-18 (panel images) + Gemma 2 (chatbot) in one system.",
    version="1.0.0",
)


# ---------- schemas ----------------------------------------------------------------
class WeatherReading(BaseModel):
    irradiation: float = Field(..., ge=0, le=1.5, description="kW/m2")
    ambient_temperature: float = Field(..., ge=-20, le=60, description="deg C")
    module_temperature: float = Field(..., ge=-20, le=90, description="deg C")
    hour: float = Field(..., ge=0, lt=24)
    plant: int = Field(1, ge=1, le=2)
    actual_power: float | None = Field(None, ge=0, description="measured kW (optional)")


class ChatRequest(BaseModel):
    message: str
    date: str | None = None
    inverter: str | None = None
    diagnosis: dict | None = None
    history: list[dict] = []


# ---------- helpers ----------------------------------------------------------------
def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _check_date(date: str | None) -> str:
    dates = monitoring.available_dates()
    if date is None:
        return dates[-1]
    if date not in dates:
        raise HTTPException(404, f"date not in data ({dates[0]} .. {dates[-1]})")
    return date


def _inverter_row(name: str, date: str) -> dict:
    f = monitoring.fleet_on(date)
    row = f[f["NAME"] == name]
    if row.empty:
        raise HTTPException(404, f"inverter {name} has no data on {date}")
    return row.iloc[0].to_dict()


# ---------- system -----------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    return {
        "api": "ok",
        "power_model": monitoring.POWER_MODEL_PATH.exists(),
        "image_model": vision.model_ready(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "chatbot_model": chatbot.pick_model(),
        "ollama_models": chatbot.available_models(),
    }


@app.get("/metrics", tags=["system"])
def metrics():
    load = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    img = load(IMAGE_METRICS_PATH)
    if img:
        img.pop("history", None)
    return {"power_model": load(POWER_METRICS_PATH), "image_model": img,
            "lstm_comparison": load(RESULTS_DIR / "lstm_comparison.json")}


@app.get("/metrics/image_history", tags=["system"])
def image_history():
    return json.loads(IMAGE_METRICS_PATH.read_text()).get("history", []) if IMAGE_METRICS_PATH.exists() else []


@app.get("/metrics/power_samples", tags=["system"])
def power_samples(n: int = 3000):
    """Random sample of test-set predictions (actual vs predicted) for plots."""
    p = RESULTS_DIR / "power_test_predictions.csv"
    df = pd.read_csv(p)
    df = df[df["IRRADIATION"] > 0.05]
    return _records(df.sample(min(n, len(df)), random_state=0))


# ---------- demo images (unseen test-set photos) ------------------------------------
def _demo_list() -> list[dict]:
    """Unseen test photos; paths are stored relative to the project root."""
    p = RESULTS_DIR / "image_test_files.json"
    items = json.loads(p.read_text()) if p.exists() else []
    return [{**d, "path": str(ROOT / d["path"])} for d in items if (ROOT / d["path"]).exists()]


@app.get("/demo/images", tags=["image model"])
def demo_images():
    return [{"id": i, "label": d["label"], "file": Path(d["path"]).name} for i, d in enumerate(_demo_list())]


@app.get("/demo/image/{idx}", tags=["image model"])
def demo_image(idx: int):
    items = _demo_list()
    if not 0 <= idx < len(items):
        raise HTTPException(404, "no such demo image")
    return FileResponse(items[idx]["path"])


# ---------- numerical model ----------------------------------------------------------
@app.post("/predict/power", tags=["numerical model"])
def predict_power(r: WeatherReading):
    """Expected AC power for given weather; if actual power is sent, also returns PR + status."""
    row = pd.DataFrame([{
        "IRRADIATION": r.irradiation, "AMBIENT_TEMPERATURE": r.ambient_temperature,
        "MODULE_TEMPERATURE": r.module_temperature,
        "HOUR_SIN": math.sin(2 * math.pi * r.hour / 24),
        "HOUR_COS": math.cos(2 * math.pi * r.hour / 24),
        "PLANT": r.plant,
    }])[FEATURES]
    expected = float(monitoring.predict_power(row)[0])
    out = {"expected_power_kw": round(expected, 2)}
    if r.actual_power is not None and expected > 1:
        pr = r.actual_power / expected
        out |= {"actual_power_kw": r.actual_power, "performance_ratio": round(pr, 3),
                "status": monitoring.status_from_pr(pr)}
    return out


@app.get("/fleet/dates", tags=["numerical model"])
def fleet_dates():
    return monitoring.available_dates()


@app.get("/fleet/summary", tags=["numerical model"])
def fleet_summary(date: str | None = None):
    return monitoring.fleet_summary(_check_date(date))


@app.get("/fleet/inverters", tags=["numerical model"])
def fleet_inverters(date: str | None = None):
    return _records(monitoring.fleet_on(_check_date(date)))


@app.get("/fleet/plant_curve", tags=["numerical model"])
def plant_curve(date: str | None = None):
    return _records(monitoring.plant_day(_check_date(date)))


@app.get("/fleet/inverter/{name}", tags=["numerical model"])
def inverter_detail(name: str, date: str | None = None):
    date = _check_date(date)
    return {
        "summary": _inverter_row(name, date),
        "day_curve": _records(monitoring.inverter_day(name, date)),
        "history": _records(monitoring.inverter_history(name)),
    }


# ---------- image model --------------------------------------------------------------
@app.post("/analyze/image", tags=["image model"])
async def analyze_image(file: UploadFile = File(...)):
    if not vision.model_ready():
        raise HTTPException(503, "image model not trained yet")
    try:
        return vision.classify(await file.read())
    except Exception as e:  # unreadable / non-image upload
        raise HTTPException(400, f"could not read image: {e}")


# ---------- integrated workflow ------------------------------------------------------
@app.post("/diagnose", tags=["integrated workflow"])
async def diagnose(inverter: str = Form(...), date: str | None = Form(None),
                   file: UploadFile | None = File(None)):
    """Full pipeline: numerical evidence (PR, loss) + image evidence (cause) -> maintenance ticket."""
    date = _check_date(date)
    row = _inverter_row(inverter, date)
    img = None
    if file is not None:
        if not vision.model_ready():
            raise HTTPException(503, "image model not trained yet")
        img = vision.classify(await file.read())
    ticket = decision.recommend(row["PR"], row["LOSS_KWH"],
                                img["label"] if img else None, img["confidence"] if img else None)
    # Kept on the server so the chatbot still knows the latest ticket after a page refresh
    LAST_DIAGNOSIS.clear()
    LAST_DIAGNOSIS.update({"inverter": inverter, "date": date, **ticket})
    return {"inverter": inverter, "date": date, "numerical": row, "image": img, "recommendation": ticket}


@app.get("/forecast", tags=["integrated workflow"])
def forecast(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, plant: int = 1):
    """Open-Meteo forecast -> expected production. Falls back to the last saved forecast when offline."""
    try:
        df, source = weather.expected_production(lat, lon, plant)
    except Exception as e:
        raise HTTPException(502, f"weather service unavailable: {e}")
    return {"source": source, "hourly": _records(df), "daily": weather.daily_totals(df)}


# ---------- chatbot ------------------------------------------------------------------
def what_to_do(diag: dict | None, inverter: str | None, date: str, fleet, arabic: bool = False) -> str:
    """One ready-made answer to "what should I do?", so the chatbot restates facts instead of guessing."""
    if diag and diag.get("image_class"):
        kwh, usd, week = diag["daily_energy_loss_kwh"], diag["daily_cost_usd"], diag["weekly_cost_if_ignored_usd"]
        if arabic:
            ar = decision.ACTIONS_AR[diag["image_class"]]
            if diag["priority"] == "None":
                return f"{diag['inverter']}: لا حاجة لأي إجراء، اللوح نظيف ويعمل بشكل طبيعي."
            return (f"بالنسبة لـ {diag['inverter']}: {ar['action']}. الفريق المسؤول: {ar['team']}. "
                    f"الأولوية: {decision.PRIORITY_AR[diag['priority']]}. السبب: {ar['cause']}. "
                    f"الخسارة {kwh:,.0f} kWh في اليوم (حوالي ${usd:,.0f})، و${week:,.0f} إذا تأخر الإصلاح أسبوعًا.")
        return (f"For {diag['inverter']} ({diag.get('date', date)}): {diag['action']}. "
                f"Assigned to: {diag['team']}. Priority: {diag['priority']}. Cause: {diag['cause']}. "
                f"It loses {kwh:,.0f} kWh per day (${usd:,.0f}), ${week:,.0f} if ignored for a week.")
    name = (diag or {}).get("inverter") or inverter or (fleet.iloc[0]["NAME"] if len(fleet) else None)
    row = fleet[fleet["NAME"] == name]
    if name is None or row.empty:
        return ("لم يتم اختيار inverter بعد. افتح صفحة Fleet Monitoring لمعرفة الأضعف." if arabic else
                "No inverter selected yet. Open the Fleet Monitoring page to find the weakest inverter.")
    r = row.iloc[0]
    if arabic:
        return (f"{name} حالته {decision.STATUS_AR[r['STATUS']]}، ويعمل بنسبة {r['PR'] * 100:.1f}% فقط من إنتاجه "
                f"المتوقع، وخسر {r['LOSS_KWH']:,.0f} kWh يوم {date}. سبب المشكلة غير معروف بعد: افتح صفحة "
                f"Diagnosis & Maintenance، وارفع صورة لألواح {name}، واضغط Run full diagnosis لمعرفة الإجراء "
                f"والفريق والأولوية.")
    return (f"{name} is {r['STATUS']} at {r['PR'] * 100:.1f}% of its expected output and lost "
            f"{r['LOSS_KWH']:,.0f} kWh on {date}. Its cause is not known yet: open the 'Diagnosis & Maintenance' "
            f"page, upload a photo of {name}'s panel string and press 'Run full diagnosis' to get the action, "
            f"the team and the priority.")


def build_context(req: ChatRequest) -> dict:
    """Facts handed to Gemma. Numbers are pre-computed in plain words to avoid misreading."""
    date = _check_date(req.date)
    s = monitoring.fleet_summary(date)
    f = monitoring.fleet_on(date)
    worst = [{
        "inverter": r.NAME, "status": r.STATUS,
        "performance_percent": round(r.PR * 100, 1),
        "energy_lost_percent": round(max(0, 1 - r.PR) * 100, 1),
        "energy_lost_kwh": round(r.LOSS_KWH, 1),
    } for r in f.head(5).itertuples()]
    ctx = {
        "selected_date": date,
        "thresholds": {"normal_if_performance_percent_at_least": PR_WARNING * 100,
                       "critical_if_performance_percent_below": PR_CRITICAL * 100},
        "fleet_today": {
            "total_inverters": s["inverters"], "normal": s["normal"], "warning": s["warning"],
            "critical": s["critical"], "actual_energy_kwh": s["actual_kwh"],
            "expected_energy_kwh": s["expected_kwh"], "energy_lost_kwh": s["loss_kwh"],
            "fleet_performance_percent": round(s["fleet_pr"] * 100, 1),
        },
        "worst_inverter_today": worst[0] if worst else None,
        "five_weakest_inverters_today_sorted_worst_first": worst,
    }
    if req.inverter:
        f_inv = f[f["NAME"] == req.inverter]
        if not f_inv.empty:  # the selected inverter may have no readings on another day
            r = f_inv.iloc[0]
            ctx["selected_inverter"] = {
                "name": req.inverter, "status": r["STATUS"],
                "performance_percent": round(r["PR"] * 100, 1),
                "actual_energy_kwh": round(r["ACTUAL_KWH"], 1), "expected_energy_kwh": round(r["EXPECTED_KWH"], 1),
            }
    diag = req.diagnosis or (dict(LAST_DIAGNOSIS) if LAST_DIAGNOSIS else None)
    arabic = any("؀" <= ch <= "ۿ" for ch in req.message)
    if diag and not arabic:  # for Arabic questions the Arabic what_to_do_now carries the ticket instead;
        ctx["latest_diagnosis"] = diag  # the English ticket text made the small model answer in English
    ctx = {"what_to_do_now": what_to_do(diag, req.inverter, date, f, arabic), **ctx}  # first, so small models see it
    m = metrics()
    if m["power_model"]:
        ctx["power_model_accuracy_R2"] = round(m["power_model"]["scores"]["RandomForest"]["R2"], 3)
    if m["image_model"]:
        ctx["image_model_test_accuracy_percent"] = round(m["image_model"]["test_accuracy"] * 100, 1)
    return ctx


@app.post("/chat/context", tags=["chatbot"])
def chat_context(req: ChatRequest):
    """Shows exactly what the chatbot receives (useful for the demo / report)."""
    return build_context(req)


@app.post("/chat", tags=["chatbot"])
def chat(req: ChatRequest):
    """Streams Gemma's answer as plain text."""
    ctx = build_context(req)
    return StreamingResponse(chatbot.stream_answer(req.message, ctx, req.history), media_type="text/plain")
