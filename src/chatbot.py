"""Transformer chatbot: Gemma 2 served locally by Ollama, grounded on live system results.

Every question is sent together with a CONTEXT block built from the numerical model,
the image model and the decision engine, so the assistant explains *this* solar site's
data instead of chatting generically.
"""
import json
import re
from collections.abc import Iterator

import requests

from config import OLLAMA_FALLBACK, OLLAMA_MODEL, OLLAMA_URL

SYSTEM_PROMPT = """You are SolarBot, the assistant of the "Smart Solar Panel Monitoring & Maintenance System".
The system has three AI parts:
1) A Random Forest model that predicts the EXPECTED power of each inverter from weather
   (irradiation, temperatures, time). Performance Ratio PR = actual / expected energy.
   PR >= 0.90 Normal, 0.75-0.90 Warning, < 0.75 Critical.
2) A ResNet image model that classifies panel photos: Clean, Dusty, Bird-drop,
   Electrical-damage, Physical-damage, Snow-Covered.
3) A decision engine that combines both into a maintenance action and priority.

Rules:
- When the user asks what to do, what action to take, or how to fix an inverter (in any language),
  reply with the text of `what_to_do_now` in 2-4 short plain sentences (no headings, no bullet
  lists). Never give generic advice when `what_to_do_now` has a specific answer.
- Never write URLs, web links or e-mail addresses. Refer to dashboard pages by their names only.
- Never mention internal field names (such as what_to_do_now or latest_diagnosis) in the answer.
- Answer ONLY from the SYSTEM DATA below and general solar-maintenance knowledge. Never invent numbers.
- If the data needed is missing, say so and tell the user which page/action provides it.
- Be concise and practical: short paragraphs or bullet points, include the key numbers.
- Reply in the same language the user writes in (Arabic or English)."""


def available_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def pick_model() -> str | None:
    models = available_models()
    for m in (OLLAMA_MODEL, OLLAMA_FALLBACK):
        if m in models:
            return m
    return None


def _norm(text: str) -> str:
    """Light Arabic normalisation so keyword matching ignores hamza forms and diacritics."""
    text = re.sub(r"[ً-ْـ]", "", text)  # tashkeel + tatweel
    return text.translate(str.maketrans("أإآىة", "ااايه")).lower()


# Retrieval step: which fact answers which kind of question (keywords are normalised Arabic)
INTENTS = {
    "action": ("اعمل", "نعمل", "نفعل", "افعل", "اسوي", "نسوي", "الحل", "الاجراء", "نصلح", "اصلح", "what should"),
    "loss": ("خسر", "خساره", "ضايع", "ضايعه", "ضايعة", "ضائع", "فقد", "كم طاقه", "قديش طاقه"),
    "worst": ("اسوا", "اضعف", "الاسوا", "الاضعف", "اقل اداء"),
    "critical": ("حرج", "حرجه", "خطر", "كم inverter", "عدد"),
}


def arabic_facts(ctx: dict, question: str) -> str:
    fleet, worst = ctx.get("fleet_today") or {}, ctx.get("worst_inverter_today") or {}
    facts = {
        "loss": (f"الطاقة الضائعة في المحطة كلها اليوم: {fleet.get('energy_lost_kwh', 0):,.0f} kWh من أصل "
                 f"{fleet.get('expected_energy_kwh', 0):,.0f} kWh متوقعة، أي "
                 f"{100 - fleet.get('fleet_performance_percent', 100):.1f}% من الطاقة المتوقعة، وأداء المحطة "
                 f"{fleet.get('fleet_performance_percent')}%.") if fleet else None,
        "critical": (f"عدد الـ inverters الحرجة اليوم {fleet.get('critical')} من {fleet.get('total_inverters')}، "
                     f"وحالة التحذير {fleet.get('warning')}.") if fleet else None,
        "worst": (f"أسوأ inverter اليوم: {worst.get('inverter')} بأداء {worst.get('performance_percent')}% "
                  f"وخسارة {worst.get('energy_lost_kwh', 0):,.0f} kWh.") if worst else None,
        "action": ctx.get("what_to_do_now"),
    }
    q = _norm(question)
    wanted = [k for k, words in INTENTS.items() if any(w in q for w in words)] or list(facts)
    lines = [f"- {facts[k]}" for k in wanted if facts.get(k)]
    return ("(معلومات من النظام تجيب عن هذا السؤال:\n" + "\n".join(lines) + "\n"
            "أجب باللغة العربية فقط، بجمل قصيرة وواضحة وبدون روابط، واعتمد على هذه المعلومات فقط.)")


def build_messages(question: str, context: dict, history: list[dict]) -> list[dict]:
    data = json.dumps(context, ensure_ascii=False, indent=1, default=str)
    system = f"{SYSTEM_PROMPT}\n\nSYSTEM DATA (live):\n{data}"
    msgs = [{"role": "system", "content": system}]
    msgs += [m for m in history[-8:] if m.get("role") in ("user", "assistant")]
    if any("؀" <= ch <= "ۿ" for ch in question):
        # Small models follow the user turn more closely than the system prompt, so the key facts
        # are repeated here in Arabic, each one labelled so the model picks the one that was asked.
        question += "\n\n" + arabic_facts(context, question)
    msgs.append({"role": "user", "content": question})
    return msgs


def stream_answer(question: str, context: dict, history: list[dict]) -> Iterator[str]:
    model = pick_model()
    if model is None:
        yield ("⚠️ Ollama is not running or no Gemma model is installed. "
               "Start Ollama and run: `ollama pull gemma2:9b`")
        return
    payload = {
        "model": model,
        "messages": build_messages(question, context, history),
        "stream": True,
        # temperature 0 + fixed seed: the same question on the same data always gets the same answer
        "options": {"temperature": 0, "seed": 42, "repeat_penalty": 1.15, "num_ctx": 4096},
    }
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "message" in chunk:
                    yield chunk["message"].get("content", "")
                if chunk.get("done"):
                    break
    except requests.RequestException as e:  # Ollama stopped or timed out mid-answer
        yield f"\n\n⚠️ The chatbot model stopped responding ({type(e).__name__}). Check that Ollama is running."
