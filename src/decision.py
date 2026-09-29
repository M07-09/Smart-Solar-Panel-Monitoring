"""Maintenance decision engine — fuses the numerical and the image model.

numerical model  -> HOW MUCH energy is lost (performance ratio, kWh lost)
image model      -> WHY it is lost (visible panel condition)
decision engine  -> WHAT to do, and HOW URGENT (action, priority, cost of waiting)
"""
from config import PR_CRITICAL, PR_WARNING

TARIFF_USD_PER_KWH = 0.10

ACTIONS = {
    "Clean": {
        "cause": "No visible defect on the panel surface",
        "action": "Inspect inverter, string wiring and connectors (electrical check)",
        "team": "Electrical technician",
        "base": 1,
    },
    "Dusty": {
        "cause": "Dust/soiling layer blocking sunlight",
        "action": "Schedule panel cleaning (water + soft brush, early morning)",
        "team": "Cleaning crew",
        "base": 2,
    },
    "Bird-drop": {
        "cause": "Bird droppings causing partial shading / hot spots",
        "action": "Spot-clean affected panels; consider bird deterrent spikes",
        "team": "Cleaning crew",
        "base": 2,
    },
    "Snow-Covered": {
        "cause": "Snow covering the panel surface",
        "action": "Remove snow with a soft roof rake or wait for melting",
        "team": "Site operator",
        "base": 2,
    },
    "Electrical-damage": {
        "cause": "Electrical damage (burn marks, hot spots, diode/junction-box failure)",
        "action": "Isolate the string and send an electrician immediately — fire risk",
        "team": "Electrician (urgent)",
        "base": 4,
    },
    "Physical-damage": {
        "cause": "Cracked or broken glass / cells",
        "action": "Replace the damaged panel; check mounting structure",
        "team": "Maintenance team + spare panel",
        "base": 3,
    },
}
PRIORITY = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}

# Arabic wording, so the chatbot can restate a ticket in clean Arabic instead of translating it
ACTIONS_AR = {
    "Clean": {"cause": "لا يوجد عيب ظاهر على سطح اللوح",
              "action": "فحص الـ inverter وتوصيلات الـ string الكهربائية", "team": "فني كهرباء"},
    "Dusty": {"cause": "طبقة غبار تحجب أشعة الشمس",
              "action": "جدولة تنظيف الألواح بالماء وفرشاة ناعمة في الصباح الباكر", "team": "فريق التنظيف"},
    "Bird-drop": {"cause": "فضلات طيور تسبب ظلًا جزئيًا ونقاطًا ساخنة",
                  "action": "تنظيف موضعي للألواح المتأثرة وتركيب طارد للطيور", "team": "فريق التنظيف"},
    "Snow-Covered": {"cause": "ثلج يغطي سطح اللوح",
                     "action": "إزالة الثلج بأداة ناعمة أو انتظار ذوبانه", "team": "مشغّل الموقع"},
    "Electrical-damage": {"cause": "ضرر كهربائي (آثار احتراق، نقاط ساخنة، أو عطل في صندوق التوصيل)",
                          "action": "افصل الـ string وأرسل كهربائيًا فورًا لأن هناك خطر حريق",
                          "team": "كهربائي (عاجل)"},
    "Physical-damage": {"cause": "زجاج أو خلايا مكسورة",
                        "action": "استبدال اللوح المتضرر وفحص هيكل التثبيت", "team": "فريق الصيانة مع لوح بديل"},
}
PRIORITY_AR = {"None": "لا يوجد", "Low": "منخفضة", "Medium": "متوسطة", "High": "عالية", "Urgent": "عاجلة"}
STATUS_AR = {"Normal": "طبيعي", "Warning": "تحذير", "Critical": "حرج"}


def recommend(pr: float | None, loss_kwh: float | None, image_class: str | None,
              confidence: float | None = None) -> dict:
    """Combine performance data and image evidence into one maintenance ticket."""
    # Score from the numerical side
    if pr is None:
        perf_score, perf_state = 0, "Unknown"
    elif pr < PR_CRITICAL:
        perf_score, perf_state = 2, "Critical"
    elif pr < PR_WARNING:
        perf_score, perf_state = 1, "Warning"
    else:
        perf_score, perf_state = 0, "Normal"

    if image_class is None:
        info = {"cause": "Not inspected yet", "action": "Upload/capture a panel image to find the cause",
                "team": "-", "base": 1}
    else:
        info = ACTIONS[image_class]

    if image_class == "Clean" and perf_score == 0:
        level, action = 0, "No action needed — panel is clean and performing as expected"
    else:
        level = min(3, max(1, info["base"] + perf_score - (1 if image_class == "Clean" else 0)))
        action = info["action"]
    if image_class in ("Electrical-damage", "Physical-damage"):
        level = max(level, info["base"])  # safety issues are never downgraded; only they reach Urgent

    daily_loss = float(loss_kwh or 0.0)
    return {
        "performance_state": perf_state,
        "performance_ratio": None if pr is None else round(pr, 3),
        "image_class": image_class,
        "image_confidence": None if confidence is None else round(confidence, 3),
        "cause": info["cause"],
        "action": action,
        "team": info["team"] if level else "-",
        "priority": PRIORITY.get(level, "None"),
        "daily_energy_loss_kwh": round(daily_loss, 1),
        "daily_cost_usd": round(daily_loss * TARIFF_USD_PER_KWH, 2),
        "weekly_cost_if_ignored_usd": round(daily_loss * TARIFF_USD_PER_KWH * 7, 2),
    }
