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
import gradio as gr

from backend.models import model_manager


model_manager.load_all()


def analyze_sentiment(text):
    if not text or not text.strip():
        return "Enter text first."
    result = model_manager.predict_sentiment(text.strip())
    return f"{result['label']} ({result['score']:.2%} confidence)"


def extract_entities(text):
    if not text or not text.strip():
        return []
    return [
        {
            "entity": entity["text"],
            "type": entity["type"],
            "confidence": round(entity["score"], 4),
            "start": entity["start"],
            "end": entity["end"],
        }
        for entity in model_manager.predict_ner(text.strip())
    ]


def answer_question(question, context):
    if not question or not question.strip() or not context or not context.strip():
        return "Enter both a question and context.", 0.0
    result = model_manager.predict_qa(question.strip(), context.strip())
    return result["answer"], result["score"]


with gr.Blocks(title="TriNLP") as demo:
    gr.Markdown("# TriNLP\nSentiment analysis, named entity recognition, and question answering.")

    with gr.Tab("Sentiment"):
        sentiment_text = gr.Textbox(label="Text", lines=4)
        sentiment_button = gr.Button("Analyze sentiment")
        sentiment_output = gr.Textbox(label="Prediction")
        sentiment_button.click(analyze_sentiment, sentiment_text, sentiment_output)

    with gr.Tab("Named Entities"):
        ner_text = gr.Textbox(label="Text", lines=5)
        ner_button = gr.Button("Extract entities")
        ner_output = gr.JSON(label="Entities")
        ner_button.click(extract_entities, ner_text, ner_output)

    with gr.Tab("Question Answering"):
        qa_context = gr.Textbox(label="Context", lines=8)
        qa_question = gr.Textbox(label="Question")
        qa_button = gr.Button("Get answer")
        qa_answer = gr.Textbox(label="Answer")
        qa_score = gr.Number(label="Confidence")
        qa_button.click(answer_question, [qa_question, qa_context], [qa_answer, qa_score])


demo.launch()
