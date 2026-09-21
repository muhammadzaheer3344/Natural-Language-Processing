# Phase 6 — TriNLP Deployment

The deployment phase: three NLP tasks (sentiment, NER, question-answering) served over
HTTP by a **FastAPI backend**, with a **Streamlit frontend** on top — nothing else in the
stack. This is where earlier phases' "trained a model in a notebook" becomes "a running
service someone can actually call."

> Unlike Phases 1–4, this phase doesn't ship its own fine-tuned weights — the backend
> pulls three pretrained models straight from the Hugging Face Hub at startup (see
> "Models used" below). That keeps the deployable image small and is a realistic pattern:
> you'd swap in your own fine-tuned checkpoint (e.g. `Phase-04_CineMood-BERT/models/cinemood_bert`)
> the same way, by pointing `models.py` at a local path instead of a Hub model name.

## Architecture

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│  Streamlit UI    │  ───────────────────▶ │  FastAPI backend  │
│  frontend/        │  ◀─────────────────── │  backend/           │
│  streamlit_app.py │      JSON              │  app.py, models.py, │
└─────────────────┘                        │  schemas.py         │
                                            └──────────────────┘
```

- **`backend/`** — FastAPI app. Three endpoints (`/sentiment`, `/ner`, `/qa`), plus
  `/health` and `/models` for introspection. Models load in the background; inference
  endpoints return `503` until `/health` reports `"models": "ready"`.
- **`frontend/`** — Streamlit app. Three tabs, one per task, calling the backend over
  `requests`. The backend URL is configurable from the sidebar (defaults to
  `http://localhost:8000`, or set `TRINLP_API_URL`).
- **`tests/`** — `pytest` tests against the FastAPI app directly (via `TestClient`, no
  server needed to run them).
- **`Dockerfile`** — containerizes the FastAPI backend.

## Models used (pulled from Hugging Face Hub)

| Task | Model |
|---|---|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| NER | `dslim/bert-base-NER` |
| QA | `deepset/bert-base-cased-squad2` |

## Running it locally

```bash
cd Phase-06_TriNLP-Deployment
pip install -r requirements.txt

# Terminal 1 — backend (models download on first run)
uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

Streamlit opens at `http://localhost:8501`; the backend serves at
`http://localhost:8000` (interactive docs at `http://localhost:8000/docs`).

If the backend is running somewhere other than `localhost:8000`, either set the
environment variable before launching Streamlit, or change the URL in the sidebar:

```bash
export TRINLP_API_URL="http://your-backend-host:8000"
streamlit run frontend/streamlit_app.py
```

For restricted or unreliable Hugging Face connectivity, download complete model
snapshots into `models/sentiment`, `models/ner`, and `models/qa`, then set
`TRINLP_MODEL_DIR` to the absolute path of the `models` directory before starting
the backend. Do not point this variable at a partially downloaded cache.

## Running the backend in Docker

```bash
docker build -t trinlp-api .
docker run -p 8000:8000 trinlp-api
```

Then run Streamlit locally against it as above (point `TRINLP_API_URL` at wherever the
container is reachable).

## API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "healthy"}` |
| GET | `/models` | — | model names in use per task |
| POST | `/sentiment` | `{"text": "..."}` | `{"label": "POSITIVE"/"NEGATIVE", "score": float}` |
| POST | `/ner` | `{"text": "..."}` | `{"entities": [{"text", "type", "start", "end", "score"}, ...]}` |
| POST | `/qa` | `{"question": "...", "context": "..."}` | `{"answer": "...", "score": float}` |

## Tests

```bash
pip install pytest
pytest tests/test_api.py -v
```

## What's inside

```
backend/
  app.py            <- FastAPI app + routes
  models.py          <- ModelManager: loads/runs the 3 Hub models
  schemas.py          <- Pydantic request/response models
frontend/
  streamlit_app.py     <- Streamlit UI (3 tabs: Sentiment, NER, QA)
tests/
  test_api.py            <- pytest suite against the FastAPI app
Dockerfile
requirements.txt
```
