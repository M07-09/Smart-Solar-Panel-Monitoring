"""Streamlit dashboard — the main interface of the Smart Solar Panel Monitoring system.

Every number on screen comes from the FastAPI backend (no model is loaded here).
Run:  streamlit run dashboard/app.py      (API must be running on :8000)
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import API_URL as API  # noqa: E402  (only the address; no model is imported here)
STATUS_COLORS = {"Normal": "#2e9e5b", "Warning": "#f0a202", "Critical": "#d7263d"}
PRIORITY_ICON = {"None": "✅", "Low": "🟢", "Medium": "🟡", "High": "🟠", "Urgent": "🔴"}

st.set_page_config(page_title="Smart Solar Monitoring", page_icon="☀️", layout="wide")
# Arabic chat messages: each paragraph picks its own direction (RTL for Arabic, LTR for English)
st.markdown("<style>[data-testid='stChatMessage'] p, [data-testid='stChatMessage'] li"
            "{unicode-bidi: plaintext; text-align: start;}</style>", unsafe_allow_html=True)


# ---------- API helpers --------------------------------------------------------------
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, **kw):
    r = requests.post(f"{API}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _health_cached():
    return api_get("/health")  # exceptions are not cached, so a failed check is retried next run


def health():
    try:
        return _health_cached()
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False)
def dates():
    return api_get("/fleet/dates")


@st.cache_data(show_spinner=False)
def fleet(date):
    return pd.DataFrame(api_get("/fleet/inverters", date=date))


@st.cache_data(show_spinner=False)
def summary(date):
    return api_get("/fleet/summary", date=date)


@st.cache_data(ttl=60, show_spinner=False)
def metrics():
    return api_get("/metrics")


@st.cache_data(ttl=900, show_spinner="Fetching live weather…")
def forecast(lat, lon, plant):
    return api_get("/forecast", lat=lat, lon=lon, plant=plant)  # weather changes slowly; cache 15 min


def status_badge(s: str) -> str:
    color = {"Normal": "green", "Warning": "orange", "Critical": "red"}.get(s, "gray")
    return f":{color}[**{s}**]"


# ---------- sidebar ------------------------------------------------------------------
h = health()
if h is None:
    st.error("🔌 The API is not running. Start it with:  `uvicorn api.main:app --port 8000`")
    if st.button("🔄 Retry"):
        st.rerun()
    st.stop()

all_dates = dates()
with st.sidebar:
    st.markdown("## ☀️ Smart Solar")
    st.caption("Panel Monitoring & Maintenance")
    date = st.selectbox("📅 Monitoring day", all_dates, index=all_dates.index("2020-05-20")
                        if "2020-05-20" in all_dates else len(all_dates) - 1)
    st.divider()
    st.markdown("**System status**")
    st.markdown(f"{'🟢' if h['api'] == 'ok' else '🔴'} FastAPI backend")
    st.markdown(f"{'🟢' if h['power_model'] else '🔴'} Random Forest (power)")
    st.markdown(f"{'🟢' if h['image_model'] else '🟡'} ResNet-18 (images)")
    st.markdown(f"{'🟢' if h['chatbot_model'] else '🔴'} Gemma chatbot "
                f"`{h['chatbot_model'] or 'offline'}`")
    st.markdown(f"{'🟢' if h['gpu'] else '⚪'} GPU: {h['gpu'] or 'CPU only'}")


# ---------- pages --------------------------------------------------------------------
def page_overview():
    st.title("☀️ Smart Solar Panel Monitoring & Maintenance System")
    st.markdown(
        "**Problem:** solar panels silently lose output because of dust, bird droppings, cracks "
        "or electrical faults. Operators usually notice only at the end of the month — after the "
        "energy (and money) is already lost, and without knowing *which* panel or *why*.\n\n"
        "**Solution:** an AI system that detects **how much** energy each inverter is losing "
        "(numerical model), finds **why** from a panel photo (image model), decides **what to do** "
        "(decision engine) and **explains it** in natural language (Transformer chatbot)."
    )
    s = summary(date)
    c = st.columns(5)
    c[0].metric("Inverters monitored", s["inverters"])
    c[1].metric("Normal", s["normal"])
    c[2].metric("Warning", s["warning"])
    c[3].metric("Critical", s["critical"])
    c[4].metric("Fleet performance", f"{s['fleet_pr'] * 100:.1f} %")
    c = st.columns(3)
    c[0].metric("Actual energy", f"{s['actual_kwh']:,.0f} kWh")
    c[1].metric("Expected energy (AI)", f"{s['expected_kwh']:,.0f} kWh")
    c[2].metric("Energy lost", f"{s['loss_kwh']:,.0f} kWh", f"≈ ${s['loss_kwh'] * 0.10:,.0f}",
                delta_color="inverse")

    st.subheader("System architecture & workflow")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#555555"];
  sensors [label="Inverter + weather\\nsensor data" fillcolor="#fff3c4"];
  photo   [label="Panel photo\\n(drone / technician)" fillcolor="#fff3c4"];
  meteo   [label="Open-Meteo\\nlive forecast" fillcolor="#fff3c4"];
  api     [label="FastAPI\\nbackend" fillcolor="#cfe8ff" shape=box3d];
  rf      [label="Random Forest\\nexpected power → PR" fillcolor="#d7f5dd"];
  cnn     [label="ResNet-18\\npanel condition" fillcolor="#d7f5dd"];
  dec     [label="Decision engine\\naction + priority + cost" fillcolor="#ffe0cc"];
  llm     [label="Gemma 2 (Transformer)\\nchatbot via Ollama" fillcolor="#ead7ff"];
  ui      [label="Streamlit\\ndashboard" fillcolor="#cfe8ff"];
  user    [label="Operator /\\nmaintenance team" shape=ellipse fillcolor="#eeeeee"];
  sensors -> api; photo -> api; meteo -> api;
  api -> rf [label="how much lost?"]; api -> cnn [label="why?"];
  rf -> dec; cnn -> dec; dec -> api;
  api -> llm [label="live results as context"]; llm -> api;
  api -> ui -> user; user -> ui [label="questions / uploads"];
}""")
    st.info("👉 Suggested demo flow: **Fleet Monitoring** (find the worst inverter) → "
            "**Diagnosis & Maintenance** (upload its panel photo) → **AI Assistant** (ask what to do).")


def page_fleet():
    st.title("📊 Fleet Monitoring — Random Forest")
    st.caption("The Random Forest predicts the power each inverter SHOULD produce under the measured "
               "weather. Performance Ratio (PR) = actual ÷ expected energy. "
               "PR ≥ 90 % Normal · 75–90 % Warning · < 75 % Critical.")
    f = fleet(date)

    curve = pd.DataFrame(api_get("/fleet/plant_curve", date=date))
    curve["DATE_TIME"] = pd.to_datetime(curve["DATE_TIME"])
    fig = go.Figure()
    for plant, dash in ((1, "solid"), (2, "dot")):
        d = curve[curve["PLANT"] == plant]
        fig.add_scatter(x=d["DATE_TIME"], y=d["EXPECTED"], name=f"Plant {plant} expected (AI)",
                        line=dict(dash=dash, color="#1f77b4"))
        fig.add_scatter(x=d["DATE_TIME"], y=d["AC_POWER"], name=f"Plant {plant} actual",
                        line=dict(dash=dash, color="#ff7f0e"))
    fig.update_layout(title=f"Plant power on {date}: actual vs AI-expected", yaxis_title="AC power (kW)",
                      height=380, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        bar = px.bar(f.sort_values("PR"), x="NAME", y=f.sort_values("PR")["PR"] * 100, color="STATUS",
                     color_discrete_map=STATUS_COLORS, labels={"y": "Performance %", "NAME": "Inverter"},
                     title="Performance ratio per inverter")
        bar.add_hline(y=90, line_dash="dash", line_color="#f0a202")
        bar.add_hline(y=75, line_dash="dash", line_color="#d7263d")
        bar.update_layout(height=380, xaxis_tickangle=-60)
        st.plotly_chart(bar, width="stretch")
    with right:
        show = f[["NAME", "STATUS", "PR", "ACTUAL_KWH", "EXPECTED_KWH", "LOSS_KWH"]].copy()
        show["PR"] = (show["PR"] * 100).round(1)
        st.dataframe(show.round(1), hide_index=True, height=380,
                     column_config={"PR": st.column_config.ProgressColumn("Perf. %", min_value=0,
                                                                          max_value=110, format="%.1f")})

    st.subheader("🔍 Inverter drill-down")
    name = st.selectbox("Inverter", f["NAME"].tolist(), key="drill")
    det = api_get(f"/fleet/inverter/{name}", date=date)
    s = det["summary"]
    c = st.columns(4)
    c[0].markdown(f"Status: {status_badge(s['STATUS'])}")
    c[1].metric("Performance", f"{s['PR'] * 100:.1f} %")
    c[2].metric("Actual energy", f"{s['ACTUAL_KWH']:,.0f} kWh")
    c[2].caption(f"Expected by the model: {s['EXPECTED_KWH']:,.0f} kWh")
    c[3].metric("Energy lost", f"{s['LOSS_KWH']:,.0f} kWh")
    a, b = st.columns(2)
    day = pd.DataFrame(det["day_curve"])
    day["DATE_TIME"] = pd.to_datetime(day["DATE_TIME"])
    g = go.Figure()
    g.add_scatter(x=day["DATE_TIME"], y=day["EXPECTED"], name="Expected (AI)", line=dict(color="#1f77b4"))
    g.add_scatter(x=day["DATE_TIME"], y=day["AC_POWER"], name="Actual", fill="tozeroy",
                  line=dict(color="#ff7f0e"))
    g.update_layout(title=f"{name} — {date}", height=340, yaxis_title="kW",
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    a.plotly_chart(g, width="stretch")
    hist = pd.DataFrame(det["history"])
    hg = px.line(hist, x="DATE", y=hist["PR"] * 100, markers=True, title=f"{name} — daily performance trend",
                 labels={"y": "Performance %"})
    hg.add_hrect(y0=0, y1=75, fillcolor="#d7263d", opacity=0.08, line_width=0)
    hg.add_hrect(y0=75, y1=90, fillcolor="#f0a202", opacity=0.08, line_width=0)
    hg.update_layout(height=320)
    b.plotly_chart(hg, width="stretch")

    with st.expander("🧪 Try the model yourself — POST /predict/power"):
        c = st.columns(6)
        irr = c[0].number_input("Irradiation kW/m²", 0.0, 1.5, 0.8, 0.05)
        amb = c[1].number_input("Ambient °C", -10.0, 55.0, 30.0)
        mod = c[2].number_input("Module °C", -10.0, 85.0, 50.0)
        hour = c[3].number_input("Hour", 0.0, 23.75, 12.0, 0.25)
        plant = c[4].selectbox("Plant", [1, 2])
        act = c[5].number_input("Actual kW (optional)", 0.0, 2000.0, 600.0)
        if st.button("Predict expected power"):
            res = api_post("/predict/power", json=dict(irradiation=irr, ambient_temperature=amb,
                           module_temperature=mod, hour=hour, plant=plant, actual_power=act))
            st.json(res)


def image_picker(key: str):
    """Upload a photo or choose an unseen test-set photo. Returns (bytes, filename, true_label)."""
    src = st.radio("Image source", ["Upload a photo", "Demo photo from test set"], horizontal=True, key=f"src_{key}")
    if src == "Upload a photo":
        up = st.file_uploader("Panel photo", type=["jpg", "jpeg", "png", "webp", "bmp"], key=f"up_{key}")
        return (up.getvalue(), up.name, None) if up else (None, None, None)
    demos = api_get("/demo/images")
    if not demos:
        st.warning("Demo images appear after the image model is trained.")
        return None, None, None
    labels = sorted({d["label"] for d in demos})
    lab = st.selectbox("True class (hidden from the model)", labels, key=f"lab_{key}")
    opts = [d for d in demos if d["label"] == lab]
    pick = st.selectbox("Photo", opts, format_func=lambda d: d["file"], key=f"pick_{key}")
    data = requests.get(f"{API}/demo/image/{pick['id']}", timeout=30).content
    return data, pick["file"], lab


def page_inspection():
    st.title("📷 Panel Inspection — ResNet-18")
    st.caption("A ResNet-18 pre-trained on ImageNet and fine-tuned on solar panel photos classifies the "
               "panel condition into 6 classes.")
    if not h["image_model"]:
        st.warning("Image model not trained yet — run `python src/train_image.py`.")
        return
    data, fname, true_label = image_picker("insp")
    if data is None:
        return
    a, b = st.columns([2, 3])
    a.image(data, caption=fname, width="stretch")
    res = api_post("/analyze/image", files={"file": (fname, data)})
    with b:
        st.subheader(f"Prediction: **{res['label']}**  ({res['confidence'] * 100:.1f} %)")
        if true_label:
            st.markdown("✅ Correct" if true_label == res["label"] else f"❌ True class: **{true_label}**")
        p = pd.Series(res["probabilities"]).sort_values()
        fig = px.bar(x=p.values * 100, y=p.index, orientation="h", labels={"x": "Probability %", "y": ""},
                     color=p.values, color_continuous_scale="Blues")
        fig.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


def page_diagnosis():
    st.title("🛠️ Diagnosis & Maintenance — full AI workflow")
    st.caption("Step 1: the Random Forest says HOW MUCH an inverter is losing.  Step 2: the ResNet says "
               "WHY from a panel photo.  Step 3: the decision engine says WHAT to do and HOW URGENT.")
    f = fleet(date)
    order = f.sort_values("PR")["NAME"].tolist()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### 1️⃣ Select inverter")
        name = st.selectbox("Inverter (worst first)", order, format_func=lambda n: (
            f"{n} — {f.set_index('NAME').loc[n, 'PR'] * 100:.0f}% "
            f"({f.set_index('NAME').loc[n, 'STATUS']})"))
        row = f.set_index("NAME").loc[name]
        st.markdown(f"Status: {status_badge(row['STATUS'])}")
        st.metric("Performance ratio", f"{row['PR'] * 100:.1f} %")
        st.metric("Energy lost today", f"{row['LOSS_KWH']:,.0f} kWh")
    with c2:
        st.markdown("#### 2️⃣ Panel photo of this inverter's string")
        data, fname, _ = image_picker("diag")
        if data:
            st.image(data, width=260)

    if st.button("🔎 Run full diagnosis", type="primary", width="stretch"):
        files = {"file": (fname, data)} if data else None
        res = api_post("/diagnose", data={"inverter": name, "date": date}, files=files)
        st.session_state["diagnosis"] = res
        st.session_state["diag_inverter"] = name

    res = st.session_state.get("diagnosis")
    if res and res["inverter"] == name:
        t = res["recommendation"]
        st.markdown("#### 3️⃣ Maintenance ticket")
        c = st.columns([1, 1.6, 1, 1])
        c[0].metric("Priority", f"{PRIORITY_ICON.get(t['priority'], '')} {t['priority']}")
        c[1].metric("Panel condition", t["image_class"] or "not inspected")
        if t["image_confidence"]:
            c[1].caption(f"{t['image_confidence'] * 100:.0f}% confidence")
        c[2].metric("Daily loss", f"${t['daily_cost_usd']:,.0f}")
        c[2].caption(f"{t['daily_energy_loss_kwh']:,.0f} kWh of energy")
        c[3].metric("Cost if ignored 1 week", f"${t['weekly_cost_if_ignored_usd']:,.0f}")
        box = st.error if t["priority"] in ("High", "Urgent") else st.warning if t["priority"] == "Medium" else st.success
        box(f"**Cause:** {t['cause']}\n\n**Action:** {t['action']}\n\n**Assigned to:** {t['team']}")
        st.caption("💬 This diagnosis is now shared with the AI Assistant — ask it to explain.")


def page_forecast():
    st.title("🌤️ Production Forecast — live weather")
    st.caption("Live hourly forecast from Open-Meteo (free, no key) → module temperature estimated with "
               "the NOCT formula → Random Forest predicts the expected output of one healthy inverter.")
    c = st.columns(3)
    lat = c[0].number_input("Latitude", -90.0, 90.0, 31.95, format="%.4f")
    lon = c[1].number_input("Longitude", -180.0, 180.0, 35.91, format="%.4f")
    plant = c[2].selectbox("Inverter type (plant)", [1, 2])
    try:
        fc = forecast(lat, lon, plant)
    except requests.RequestException as e:
        st.error(f"Weather service unavailable: {e}")
        return
    if fc["source"] != "live":
        st.warning(f"No internet connection to Open-Meteo, so the last forecast downloaded for this location "
                   f"is shown ({fc['source']}).")
    cols = st.columns(len(fc["daily"]))
    for col, d in zip(cols, fc["daily"]):
        col.metric(f"📅 {d['DATE']} — expected energy per inverter", f"{d['EXPECTED_KWH']:,.0f} kWh")
        col.caption(f"Peak {d['PEAK_KW']:,.0f} kW · cloud {d['AVG_CLOUD']:.0f}% · max {d['MAX_TEMP']:.0f}°C")
    hr = pd.DataFrame(fc["hourly"])
    hr["DATE_TIME"] = pd.to_datetime(hr["DATE_TIME"])
    fig = go.Figure()
    fig.add_bar(x=hr["DATE_TIME"], y=hr["EXPECTED_KW"], name="Expected power (kW)", marker_color="#f5b700")
    fig.add_scatter(x=hr["DATE_TIME"], y=hr["GHI_W_M2"], name="Solar irradiance (W/m²)", yaxis="y2",
                    line=dict(color="#1f77b4"))
    fig.update_layout(height=420, yaxis=dict(title="kW"), legend=dict(orientation="h"),
                      yaxis2=dict(title="W/m²", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Use: plan cleaning on low-production days so no sunny hours are wasted.")


def page_chat():
    st.title("💬 AI Assistant — Gemma 2 (Transformer)")
    diag = st.session_state.get("diagnosis")
    inv = st.session_state.get("diag_inverter")
    # The ticket travels with its own inverter and date, so it stays correct if the sidebar day changes
    diag_ctx = {"inverter": diag["inverter"], "date": diag["date"], **diag["recommendation"]} if diag else None
    st.caption(f"Grounded on live system data for **{date}**"
               + (f" and the latest diagnosis of **{inv}** ({diag['date']})" if diag else "")
               + f". Model: `{h['chatbot_model'] or 'offline'}`")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    with st.expander("🔎 What the chatbot receives from the API (context)"):
        st.json(api_post("/chat/context", json={"message": "", "date": date, "inverter": inv,
                                                 "diagnosis": diag_ctx}))
    examples = ["Which inverter is the worst today and how much energy did we lose?",
                "شو لازم أعمل بالـ inverter اللي شخّصته؟",
                "Explain how the system decides the maintenance priority."]
    ex_cols = st.columns(len(examples))
    clicked = None
    for c, e in zip(ex_cols, examples):
        if c.button(e, width="stretch"):
            clicked = e
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    q = st.chat_input("Ask about panels, predictions, images or maintenance…") or clicked
    if q:
        st.chat_message("user").markdown(q)
        payload = {"message": q, "date": date, "inverter": inv, "diagnosis": diag_ctx,
                   "history": st.session_state.chat}
        with st.chat_message("assistant"):
            try:
                with requests.post(f"{API}/chat", json=payload, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    r.encoding = "utf-8"
                    answer = st.write_stream(r.iter_content(chunk_size=None, decode_unicode=True))
            except requests.RequestException as e:
                answer = f"⚠️ Could not reach the chatbot: {e}"
                st.error(answer)
        st.session_state.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
    if st.session_state.chat and st.button("🗑️ Clear chat"):
        st.session_state.chat = []
        st.rerun()


def page_models():
    st.title("📈 Model Performance & API")
    m = metrics()
    pm = m["power_model"]
    st.header("Random Forest — expected power")
    if pm:
        s = pm["scores"]
        c = st.columns(4)
        c[0].metric("R² (test)", f"{s['RandomForest']['R2']:.3f}")
        c[1].metric("MAE (test)", f"{s['RandomForest']['MAE']:.1f} kW")
        c[2].metric("R² daylight", f"{s['RandomForest_daylight']['R2']:.3f}")
        c[3].metric("MAE vs Linear Reg.", f"{s['RandomForest']['MAE']:.1f} vs {s['LinearRegression']['MAE']:.1f}")
        comp = pd.DataFrame(s).T.round(3)
        a, b = st.columns(2)
        a.dataframe(comp)
        imp = pd.Series(pm["feature_importance"])
        b.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Feature importance",
                              labels={"x": "", "y": ""}).update_layout(height=300), width="stretch")
        smp = pd.DataFrame(api_get("/metrics/power_samples"))
        sc = px.scatter(smp, x="AC_POWER", y="PREDICTED", opacity=0.35,
                        title=f"Actual vs predicted (test days from {pm['test_from']})",
                        labels={"AC_POWER": "Actual kW", "PREDICTED": "Predicted kW"})
        mx = float(smp[["AC_POWER", "PREDICTED"]].max().max())
        sc.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="perfect", line=dict(dash="dash", color="gray"))
        st.plotly_chart(sc, width="stretch")

    lc = m.get("lstm_comparison")
    if lc:
        st.subheader("Why Random Forest? — comparison on the same test rows")
        rows = {k: lc[k] for k in ("RandomForest", "LinearRegression", "LSTM",
                                   "RandomForest_daylight", "LinearRegression_daylight", "LSTM_daylight")
                if k in lc}
        st.dataframe(pd.DataFrame(rows).T.round(3))
        st.caption(f"LSTM (2 layers, 8×15-min weather history) trained in {lc['lstm_train_seconds']} s on "
                   f"{lc['device']}. Power depends on the *current* irradiance, so extra history does not help; "
                   "the Random Forest is more accurate, faster and easier to explain.")

    st.header("ResNet-18 — panel condition")
    im = m["image_model"]
    if im:
        c = st.columns(4)
        c[0].metric("Test accuracy", f"{im['test_accuracy'] * 100:.1f} %")
        c[1].metric("Macro F1", f"{im['test_f1_macro']:.3f}")
        c[2].metric("Training time", f"{im['train_seconds']:.0f} s")
        c[3].metric("Device", im["device"].replace("NVIDIA GeForce ", "").replace(" Laptop GPU", ""))
        a, b = st.columns(2)
        cm = pd.DataFrame(im["confusion_matrix"], index=im["classes"], columns=im["classes"])
        a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 labels=dict(x="Predicted", y="True"), title="Confusion matrix (test)")
                       .update_layout(height=420), width="stretch")
        hist = pd.DataFrame(api_get("/metrics/image_history"))
        hl = hist.melt(id_vars=["epoch"], value_vars=["train_acc", "val_acc"])
        b.plotly_chart(px.line(hl, x="epoch", y="value", color="variable", markers=True,
                               title="Training curve (accuracy)").update_layout(height=420),
                       width="stretch")
        rep = pd.DataFrame(im["report"]).T.drop(columns=["support"], errors="ignore").round(3)
        st.dataframe(rep)
    else:
        st.info("Image model not trained yet.")

    st.header("API endpoints")
    st.markdown("Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
    st.dataframe(pd.DataFrame([
        ("GET", "/health", "Status of API, models, GPU and chatbot"),
        ("POST", "/predict/power", "Weather → expected power (+ PR if actual sent)"),
        ("GET", "/fleet/summary", "Daily fleet KPIs"),
        ("GET", "/fleet/inverters", "PR + status for every inverter"),
        ("GET", "/fleet/inverter/{name}", "Day curve + history for one inverter"),
        ("POST", "/analyze/image", "Panel photo → condition class + probabilities"),
        ("POST", "/diagnose", "Inverter + photo → maintenance ticket (full workflow)"),
        ("GET", "/forecast", "Open-Meteo forecast → expected production"),
        ("POST", "/chat", "Question + live context → Gemma answer (streamed)"),
        ("GET", "/metrics", "Model evaluation results"),
    ], columns=["Method", "Endpoint", "Purpose"]), hide_index=True, width="stretch")


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon="🏠", default=True),
    st.Page(page_fleet, title="Fleet Monitoring", icon="📊", url_path="fleet"),
    st.Page(page_inspection, title="Panel Inspection", icon="📷", url_path="inspection"),
    st.Page(page_diagnosis, title="Diagnosis & Maintenance", icon="🛠️", url_path="diagnosis"),
    st.Page(page_forecast, title="Production Forecast", icon="🌤️", url_path="forecast"),
    st.Page(page_chat, title="AI Assistant", icon="💬", url_path="assistant"),
    st.Page(page_models, title="Model Performance", icon="📈", url_path="models"),
])
pg.run()
