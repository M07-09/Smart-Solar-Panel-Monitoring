"""Capture full-page screenshots of the running system with headless Edge (Chrome DevTools Protocol).

Requires the API (:8000) and dashboard (:8501) to be running:  python run.py
Walks the demo flow in ONE Streamlit session so the diagnosis reaches the chatbot page.
"""
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import websocket

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
UI = "http://localhost:8501"
OUT = Path(__file__).resolve().parent / "screenshots"
OUT.mkdir(exist_ok=True)
W = 1440


class Browser:
    def __init__(self):
        self.proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}",
                                      f"--user-data-dir={tempfile.mkdtemp()}", "--hide-scrollbars",
                                      f"--window-size={W},900", "about:blank"])
        for _ in range(30):
            try:
                tabs = requests.get(f"http://127.0.0.1:{PORT}/json", timeout=2).json()
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        self.i = 0
        self.size(900)

    def cmd(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.i:
                return msg.get("result", {})

    def js(self, expr):
        r = self.cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def size(self, h):
        self.cmd("Emulation.setDeviceMetricsOverride", width=W, height=int(h), deviceScaleFactor=1, mobile=False)

    def goto(self, url):
        self.size(900)
        self.cmd("Page.navigate", url=url)

    def idle(self, must_contain: str, extra=3.0, timeout=90):
        """Wait until the page text contains `must_contain` and Streamlit is no longer running."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok = self.js(f"""(() => {{
                const t = document.body ? document.body.innerText : '';
                const running = !!document.querySelector('[data-testid="stStatusWidget"] img, .stSpinner');
                return t.includes({json.dumps(must_contain)}) && !running;
            }})()""")
            if ok:
                time.sleep(extra)
                return
            time.sleep(1)
        print("  ! timeout waiting for", must_contain)

    def click(self, text, selector="button, label, a, span, p, li, div[role=option]"):
        """Real mouse click on the smallest visible element whose text equals `text`."""
        rect = self.js(f"""(() => {{
            const els = [...document.querySelectorAll({json.dumps(selector)})]
              .filter(e => e.innerText && e.innerText.trim() === {json.dumps(text)} && e.offsetParent);
            if (!els.length) return null;
            const e = els[els.length - 1]; e.scrollIntoView({{block: 'center'}});
            const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2];
        }})()""")
        if rect is None:
            print("  ! element not found:", text)
            return False
        for t in ("mousePressed", "mouseReleased"):
            self.cmd("Input.dispatchMouseEvent", type=t, x=rect[0], y=rect[1], button="left", clickCount=1)
        time.sleep(1.0)
        return True

    def select(self, label, option):
        """Open the Streamlit selectbox with `label` and pick `option`."""
        rect = self.js(f"""(() => {{
            const box = [...document.querySelectorAll('[data-testid="stSelectbox"]')]
              .find(b => b.textContent.includes({json.dumps(label)}));
            const c = box && box.querySelector('button[aria-haspopup="listbox"]');
            if (!c) return null; c.scrollIntoView({{block: 'center'}});
            const r = c.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2];
        }})()""")
        if rect is None:
            print("  ! selectbox not found:", label)
            return
        for t in ("mousePressed", "mouseReleased"):
            self.cmd("Input.dispatchMouseEvent", type=t, x=rect[0], y=rect[1], button="left", clickCount=1)
        time.sleep(1.0)
        self.click(option, "[role=option], [role=option] *")

    def shot(self, name, full=True):
        self.js("window.scrollTo(0,0); document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0,0)")
        if full:
            h = self.js("""(() => {
                const m = document.querySelector('[data-testid="stMainBlockContainer"]') ||
                          document.querySelector('[data-testid="stMain"]');
                return m ? m.scrollHeight + 140 : document.body.scrollHeight; })()""") or 900
            self.size(max(900, min(h, 6000)))
            time.sleep(2.5)  # let plotly re-layout for the new height
        data = self.cmd("Page.captureScreenshot", format="png")["data"]
        (OUT / f"{name}.png").write_bytes(base64.b64decode(data))
        print("saved", name)
        self.size(900)
        time.sleep(1)

    def close(self):
        self.ws.close()
        self.proc.terminate()


def main():
    b = Browser()
    try:
        b.goto(UI)
        b.idle("System architecture", extra=5)
        b.shot("01_overview")

        b.click("Fleet Monitoring", "a span, a")
        b.idle("Inverter drill-down", extra=6)
        b.shot("02_fleet_monitoring")

        b.click("Panel Inspection", "a span, a")
        b.idle("Image source")
        b.click("Demo photo from test set", "label, p")
        b.idle("Prediction:")
        b.select("True class", "Electrical-damage")
        b.idle("Prediction:", extra=4)
        b.shot("03_panel_inspection")

        b.click("Diagnosis & Maintenance", "a span, a")
        b.idle("Select inverter")
        b.click("Demo photo from test set", "label, p")
        time.sleep(3)
        b.select("True class", "Electrical-damage")
        time.sleep(3)
        b.click("🔎 Run full diagnosis", "button, p")
        b.idle("Maintenance ticket", extra=4)
        b.shot("04_diagnosis")

        b.click("AI Assistant", "a span, a")
        b.idle("What the chatbot receives")
        b.click("Which inverter is the worst today and how much energy did we lose?", "button, p")
        b.idle("Clear chat", extra=3, timeout=180)
        b.click("شو لازم أعمل بالـ inverter اللي شخّصته؟", "button, p")
        time.sleep(4)
        b.idle("Clear chat", extra=4, timeout=180)
        b.shot("05_ai_assistant")

        b.click("Production Forecast", "a span, a")
        b.idle("expected energy per inverter", extra=5)
        b.shot("06_forecast")

        b.click("Model Performance", "a span, a")
        b.idle("API endpoints", extra=6)
        b.shot("07_model_performance")

        b.goto("http://127.0.0.1:8000/docs")
        time.sleep(6)
        b.shot("08_api_docs", full=False)
        b.size(1500)
        time.sleep(1)
        data = b.cmd("Page.captureScreenshot", format="png")["data"]
        (OUT / "08_api_docs.png").write_bytes(base64.b64decode(data))
    finally:
        b.close()


if __name__ == "__main__":
    main()
