"""
TriNLP API — End-to-End Sentiment + NER + QA
FastAPI application with three NLP endpoints.
"""

import threading

from fastapi import FastAPI, HTTPException

from schemas import (
    SentimentRequest, SentimentResponse,
    NERRequest, NERResponse, Entity,
    QARequest, QAResponse
)
from models import model_manager


# Initialize FastAPI
app = FastAPI(
    title="TriNLP API",
    description="End-to-End Sentiment + NER + QA API",
    version="1.0.0"
)


model_status = {
    "state": "loading",
    "error": None,
}


def load_models_in_background():
    try:
        model_manager.load_all()
        model_status["state"] = "ready"
    except Exception as exc:
        model_status["state"] = "error"
        model_status["error"] = str(exc)
        print(f"Model loading failed: {exc}")


def require_models():
    if model_status["state"] == "loading":
        try:
            model_manager.load_all()
            model_status["state"] = "ready"
            model_status["error"] = None
            return
        except Exception as exc:
            model_status["state"] = "error"
            model_status["error"] = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Models failed to load: {model_status['error']}",
            ) from exc
    if model_status["state"] == "error":
        raise HTTPException(
            status_code=503,
            detail=f"Models failed to load: {model_status['error']}",
        )


# Load models before the app starts serving requests so inference endpoints do not
# race with a still-loading model registry during tests or early startup.
@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_all()
        model_status["state"] = "ready"
    except Exception as exc:
        model_status["state"] = "error"
        model_status["error"] = str(exc)
        print(f"Model loading failed: {exc}")


# ============================================================
# INFO ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "name": "TriNLP API",
        "version": "1.0.0",
        "tasks": ["sentiment", "ner", "qa"],
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": model_status["state"],
        "error": model_status["error"],
    }


@app.get("/models")
def list_models():
    return {
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "ner": "dslim/bert-base-NER",
        "qa": "deepset/bert-base-cased-squad2"
    }


# ============================================================
# TASK ENDPOINTS
# ============================================================

@app.post("/sentiment", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """Classify sentiment as POSITIVE or NEGATIVE."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    require_models()
    result = model_manager.predict_sentiment(request.text)
    return SentimentResponse(**result)


@app.post("/ner", response_model=NERResponse)
def predict_ner(request: NERRequest):
    """Extract named entities (PER, ORG, LOC, MISC)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    require_models()
    entities_data = model_manager.predict_ner(request.text)
    entities = [Entity(**e) for e in entities_data]
    return NERResponse(entities=entities)


@app.post("/qa", response_model=QAResponse)
def predict_qa(request: QARequest):
    """Answer a question given a context."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not request.context.strip():
        raise HTTPException(status_code=400, detail="Context cannot be empty")

    require_models()
    result = model_manager.predict_qa(request.question, request.context)
    return QAResponse(**result)
