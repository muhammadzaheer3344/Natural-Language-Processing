"""
TriNLP API Tests
Run with: pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TriNLP API"
    assert "sentiment" in data["tasks"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "ner" in data
    assert "qa" in data


def test_sentiment_positive(client):
    response = client.post("/sentiment", json={"text": "This is amazing!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["score"] > 0.5


def test_sentiment_negative(client):
    response = client.post("/sentiment", json={"text": "This is terrible."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"


def test_sentiment_empty(client):
    response = client.post("/sentiment", json={"text": ""})
    assert response.status_code == 400


def test_ner(client):
    response = client.post("/ner", json={"text": "Elon Musk founded SpaceX."})
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) > 0


def test_ner_empty(client):
    response = client.post("/ner", json={"text": ""})
    assert response.status_code == 400


def test_qa(client):
    response = client.post("/qa", json={
        "question": "What is the capital of France?",
        "context": "France is in Europe. Its capital is Paris."
    })
    assert response.status_code == 200
    data = response.json()
    assert "Paris" in data["answer"]


def test_qa_empty_question(client):
    response = client.post("/qa", json={
        "question": "",
        "context": "Some context"
    })
    assert response.status_code == 400


def test_qa_empty_context(client):
    response = client.post("/qa", json={
        "question": "Some question?",
        "context": ""
    })
    assert response.status_code == 400
