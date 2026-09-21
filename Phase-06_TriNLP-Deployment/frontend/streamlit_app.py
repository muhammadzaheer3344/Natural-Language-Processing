"""
TriNLP — Streamlit Frontend
A thin UI over the TriNLP FastAPI backend (sentiment + NER + QA).
No other frameworks involved — just Streamlit talking to FastAPI over HTTP.

Run:
    streamlit run frontend/streamlit_app.py

The backend must be running separately:
    uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("TRINLP_API_URL", "http://localhost:8000")

st.set_page_config(page_title="TriNLP", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Sidebar — backend connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Backend")
    api_url = st.text_input("FastAPI URL", value=API_URL)
    st.caption("Change this if the backend runs on a different host/port.")

    if st.button("Check connection"):
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            if resp.status_code == 200:
                st.success("Backend is reachable ✅")
            else:
                st.error(f"Backend returned {resp.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach backend: {exc}")

st.title("🧠 TriNLP")
st.caption("Sentiment · NER · Question Answering — served by a FastAPI backend")

tab_sentiment, tab_ner, tab_qa = st.tabs(["Sentiment", "Named Entities", "Question Answering"])

# ---------------------------------------------------------------------------
# Sentiment tab
# ---------------------------------------------------------------------------
with tab_sentiment:
    st.subheader("Sentiment analysis")
    text = st.text_area(
        "Text",
        placeholder="Type or paste a sentence...",
        key="sentiment_text",
    )
    if st.button("Analyze sentiment", key="sentiment_btn"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                resp = requests.post(f"{api_url}/sentiment", json={"text": text}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    label = data["label"]
                    score = data["score"]
                    emoji = "🟢" if label == "POSITIVE" else "🔴"
                    st.metric(f"{emoji} Prediction", label, f"{score:.2%} confidence")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

# ---------------------------------------------------------------------------
# NER tab
# ---------------------------------------------------------------------------
with tab_ner:
    st.subheader("Named entity recognition")
    ner_text = st.text_area(
        "Text",
        placeholder="e.g. Elon Musk founded SpaceX in California.",
        key="ner_text",
    )
    if st.button("Extract entities", key="ner_btn"):
        if not ner_text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                resp = requests.post(f"{api_url}/ner", json={"text": ner_text}, timeout=30)
                if resp.status_code == 200:
                    entities = resp.json()["entities"]
                    if not entities:
                        st.info("No entities found.")
                    else:
                        for e in entities:
                            st.write(f"**{e['text']}** — `{e['type']}` (confidence: {e['score']:.2%})")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

# ---------------------------------------------------------------------------
# QA tab
# ---------------------------------------------------------------------------
with tab_qa:
    st.subheader("Question answering")
    context = st.text_area(
        "Context",
        placeholder="Paste a paragraph the question will be answered from...",
        key="qa_context",
    )
    question = st.text_input("Question", key="qa_question")
    if st.button("Get answer", key="qa_btn"):
        if not context.strip() or not question.strip():
            st.warning("Please fill in both the context and the question.")
        else:
            try:
                resp = requests.post(
                    f"{api_url}/qa",
                    json={"question": question, "context": context},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(data["answer"])
                    st.caption(f"Confidence: {data['score']:.2%}")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")
