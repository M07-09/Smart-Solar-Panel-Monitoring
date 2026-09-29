"""One-command launcher for the whole system:  python run.py

1. starts the FastAPI backend on http://127.0.0.1:8000 and waits until it answers /health
2. starts the Streamlit dashboard on http://localhost:8501 (the browser opens automatically)
Ctrl+C stops both.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API_PORT, UI_PORT = 8000, 8501


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_api(proc: subprocess.Popen, timeout: int = 120) -> dict | None:
    """Poll /health until the API answers; None if it exits or times out."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:  # uvicorn crashed during start-up
            return None
        try:
            return requests.get(f"http://127.0.0.1:{API_PORT}/health", timeout=15).json()
        except requests.RequestException:
            time.sleep(1)
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress messages immediately
    for port, name in ((API_PORT, "API"), (UI_PORT, "dashboard")):
        if port_busy(port):
            sys.exit(f"Port {port} ({name}) is already in use. The system may already be running; "
                     "close the other terminal that runs it, then try again.")

    print("Starting the FastAPI backend (loading the models) ...")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(API_PORT)], cwd=ROOT)
    health = wait_for_api(api)
    if health is None:
        api.terminate()
        sys.exit("The API did not start. Read the error above.")

    print(f"API ready            -> http://127.0.0.1:{API_PORT}/docs")
    print(f"GPU                  -> {health['gpu'] or 'not available (CPU)'}")
    if health["chatbot_model"]:
        print(f"Chatbot model        -> {health['chatbot_model']}")
    else:
        print("Chatbot              -> OFFLINE: start Ollama and run `ollama pull gemma2:9b`")

    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                           "--server.port", str(UI_PORT), "--server.address", "localhost"], cwd=ROOT)
    print(f"\nDashboard            -> http://localhost:{UI_PORT}\n(press Ctrl+C to stop)\n")
    try:
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()
        print("Stopped.")


if __name__ == "__main__":
    main()
