# ☀️ Smart Solar Panel Monitoring & Maintenance System

An integrated AI system that finds **which** solar inverter is losing energy, **why** it is
losing it, **what** to do about it and **how urgent** it is — and explains everything in
natural language.

| Question | Component | Model |
|---|---|---|
| How much energy is being lost? | Numerical model | **Random Forest** (vs. Linear Regression and LSTM) |
| Why? | Image model | **ResNet-18** (transfer learning, RTX 5060) |
| What should we do, and how urgent is it? | Decision engine | Rules fusing both models |
| Can you explain it? | Chatbot | **Gemma 2** Transformer via Ollama, grounded on live results |
| How do the parts connect? | API | **FastAPI** |
| Where does the user work? | Dashboard | **Streamlit** (7 pages) |

## Workflow

```
weather + inverter data ─► Random Forest ─► expected power ─► Performance Ratio (actual/expected)
                                                                   │ Warning < 90 %, Critical < 75 %
panel photo ─────────────► ResNet-18 ─────► condition (6 classes)  │
                                                                   ▼
                                      Decision engine ─► action · team · priority · $ lost
                                                                   │
            Open-Meteo forecast ─► Random Forest ─► tomorrow's expected production
                                                                   │
                     all live results ─► Gemma 2 chatbot ─► answers the operator
          (FastAPI connects every step · Streamlit shows everything)
```

## Results

| Model | Test metric |
|---|---|
| Random Forest (expected power) | **R² = 0.973**, MAE = 17.2 kW on the last 7 days |
| Same 27,146 test rows | MAE: Random Forest **16.5** · Linear Regression 23.4 · LSTM 23.2 kW, so Random Forest was chosen |
| ResNet-18 (6 classes, 133 unseen images) | **Accuracy = 90.2 %**, macro F1 = 0.902 |

Clean (29/29), Snow-Covered and Electrical-damage are recognised best. The most common mistake is
Dusty panels predicted as Clean (4/29), since a light dust layer is hard to see in some photos.
Moving from 224 px / 16 fine-tune epochs to 288 px / 26 epochs raised accuracy from 84.2 % to 90.2 %
(the first run's metrics are kept in `results/experiments/`).

## Quick start

```bash
# 1. PyTorch with CUDA (RTX 50-series needs CUDA 12.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 2. Chatbot model (Ollama must be installed and running)
ollama pull gemma2:9b          # gemma2:2b works as a lighter fallback

# 3. Train (data is downloaded from Kaggle automatically)
cd src
python data_prep.py            # clean + merge generation & weather data
python train_numeric.py        # Random Forest + Linear Regression baseline
python train_lstm.py           # LSTM + Linear Regression comparison on the same rows
python train_image.py          # ResNet-18 on the GPU (~4 min)
cd ..

# 4. Run everything (API :8000 + dashboard :8501); Ctrl+C stops both
python run.py
```

Dashboard: **http://localhost:8501** · API docs: http://127.0.0.1:8000/docs

> **Windows note:** if `import torch` fails with `WinError 4551 — An Application Control
> policy has blocked this file`, Windows **Smart App Control** is blocking PyTorch's DLLs.
> Turn it off in Windows Security → App & browser control → Smart App Control.

## Project structure

```
Smart_Solar_Monitoring/
├── api/main.py            FastAPI backend: every endpoint, connects all models
├── dashboard/app.py       Streamlit dashboard (main user interface)
├── src/
│   ├── config.py          paths, thresholds, model settings
│   ├── data_prep.py       load / clean / merge Kaggle CSVs, flag healthy rows
│   ├── train_numeric.py   Random Forest + Linear Regression
│   ├── train_lstm.py      LSTM comparison
│   ├── train_image.py     ResNet-18 transfer learning
│   ├── monitoring.py      daily performance ratio per inverter
│   ├── vision.py          image inference
│   ├── decision.py        maintenance decision engine
│   ├── weather.py         Open-Meteo live forecast
│   └── chatbot.py         Gemma 2 via Ollama, grounded prompt
├── models/                trained models (power_rf.joblib, panel_resnet.pt)
├── results/               metrics JSON, predictions, test image list
├── data/                  raw + processed data (not committed)
├── report/                final report (PDF + Word) and the scripts that build it
├── run.py                 one-command launcher
└── requirements.txt
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Status of the API, models, GPU and chatbot |
| POST | `/predict/power` | Weather → expected power (+ PR and status if actual power is sent) |
| GET | `/fleet/summary` · `/fleet/inverters` · `/fleet/inverter/{name}` | Monitoring results |
| POST | `/analyze/image` | Panel photo → condition + probabilities |
| POST | `/diagnose` | Inverter + photo → maintenance ticket (full workflow) |
| GET | `/forecast` | Live weather → expected production for today and tomorrow (uses the last saved forecast if offline) |
| POST | `/chat` · `/chat/context` | Gemma answer (streamed) · the context it receives |
| GET | `/metrics` | Model evaluation results |

## Report

`report/Smart_Solar_Monitoring_Report.pdf` and `.docx` are both built from `report/report_content.py`,
so they contain the same text. To rebuild them after a change (Edge and MS Word must be installed):

```bash
cd report
python make_figures.py && python make_architecture.py   # charts from results/
python take_screenshots.py                              # needs `python run.py` running
python build_pdf.py                                     # PDF via headless Edge
python build_docx.py                                    # Word (python-docx; Word fills the contents)
```

## Data sources

* **Solar Power Generation Data** — Kaggle (`anikannal/solar-power-generation-data`): two plants
  in India, 44 inverters, readings every 15 min for 34 days (inverter power plus irradiation and
  ambient/module temperature).
* **Solar Panel Images: Clean and Faulty** — Kaggle (`pythonafroz/solar-panel-images`): 885
  photos in 6 classes: Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow-Covered.
* **Open-Meteo** — free live weather forecast API, no key needed.
