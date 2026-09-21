import os
import subprocess
import sys
import threading
import time

import uvicorn


os.environ.setdefault("TRINLP_API_URL", "http://127.0.0.1:8000")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def run_api():
    uvicorn.run("app:app", app_dir="backend", host="127.0.0.1", port=8000)


api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

# Give FastAPI time to begin loading models while Streamlit starts its UI.
time.sleep(2)

subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/streamlit_app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "7860",
        "--server.headless",
        "true",
    ],
    check=True,
)
