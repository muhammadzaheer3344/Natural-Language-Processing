import os
import sys
import threading

import streamlit as st
import uvicorn


os.environ.setdefault("TRINLP_API_URL", "http://127.0.0.1:8000")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@st.cache_resource
def start_backend():
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    sys.path.insert(0, backend_dir)

    thread = threading.Thread(
        target=uvicorn.run,
        args=("app:app",),
        kwargs={
            "app_dir": backend_dir,
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "warning",
        },
        daemon=True,
    )
    thread.start()
    return thread


start_backend()
exec(compile(open("frontend/streamlit_app.py", encoding="utf-8").read(), "frontend/streamlit_app.py", "exec"))
