# NLP — Learning Journey

A phase-by-phase progression through NLP techniques, each phase applying a harder
technique to a concrete problem — from classical ML all the way to a served,
production-shaped API with a UI.

## Phases

| Phase | Folder | Problem | Technique | Result |
|---|---|---|---|---|
| 1 | [`Phase-01_SMS-Shield-Classical-ML`](Phase-01_SMS-Shield-Classical-ML/) | SMS spam detection | TF-IDF + classical ML (Naive Bayes / Logistic Regression / SVM) | SVM, 98.58% test accuracy |
| 2 | [`Phase-02_SMS-Shield-Neural-Network`](Phase-02_SMS-Shield-Neural-Network/) | SMS spam detection (same problem, revisited) | Keras embedding + dense NN | 98.06% test accuracy |
| 3 | [`Phase-03_CineMood-LSTM-GRU`](Phase-03_CineMood-LSTM-GRU/) | Movie review sentiment (IMDB) | Sequence models: LSTM / GRU / BiLSTM | GRU, 86.89% test accuracy |
| 4 | [`Phase-04_CineMood-BERT`](Phase-04_CineMood-BERT/) | Movie review sentiment (IMDB, same problem) | Fine-tuned DistilBERT (transformer) | 91.48% test accuracy |
| 5 | [`Phase-05_DocuMind-RAG`](Phase-05_DocuMind-RAG/) | Document question-answering | Retrieval-Augmented Generation (Chroma vector DB + embeddings) | Vector store + chunk artifacts |
| 6 | [`Phase-06_TriNLP-Deployment`](Phase-06_TriNLP-Deployment/) | Serving NLP models | Streamlit Cloud app | Working UI (sentiment, NER, QA) |

## The arc

The same underlying idea — text classification / understanding — gets progressively
harder tooling thrown at it:

```
TF-IDF + classical ML  →  learned embeddings + NN  →  sequence models (LSTM/GRU)
        →  pretrained transformers (BERT)  →  retrieval-augmented generation
          → deployment (Streamlit Cloud)
```

Phases 1–2 and 3–4 are deliberately paired: same dataset, same problem, two techniques,
so the jump in capability (and complexity) is directly comparable.

## How each phase folder is organized

Every phase folder (1–5) follows the same shape:

```
Phase-0X_Name/
  README.md          <- what this phase covers, dataset, approach, results
  <notebook>.ipynb    <- the training/experiment notebook
  models/              <- saved model artifacts (weights, tokenizer, vectorizer, etc.)
```

Phase 6 is structured differently because it's a deployable app, not a notebook — see
its own README for how to run it.

## Environment

All notebooks were built and run in **Google Colab**. Re-running a notebook end-to-end
regenerates the artifacts already saved under each phase's `models/` folder — you don't
need to re-run anything just to use the saved models.
