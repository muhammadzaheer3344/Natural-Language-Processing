# Phase 5 — DocuMind: RAG

Retrieval-Augmented Generation over a small document corpus (HR/policy, API docs, and
process PDFs) — the ingestion + retrieval half of a document question-answering system.

> **Note:** this export only contains the built **artifacts** (vector store, embeddings,
> chunk records) — the notebook/pipeline script that produced them wasn't part of this
> export. The pipeline shape below is reconstructed from `corpus_metadata.json` and the
> chunk records themselves; drop your ingestion notebook into this folder if you have it.

## Corpus

| | |
|---|---|
| Documents | 13 PDFs across 3 batches (`data-day4`, `data-day8`, `data-day10`) |
| Total characters | 22,392 |
| Total chunks | 76 |
| Chunk size / overlap | 400 chars / 50 chars |

Document mix: remote-work policies (two versions — 2023 and its 2024 replacement),
meeting notes, project updates, API reference + error codes + CLI reference + release
notes, pricing/SKU guide, refund policy, vendor onboarding, and a data-retention policy.
The two remote-work policy versions in particular make this a good corpus for testing
whether retrieval correctly prefers the current (2024) policy over the superseded one.

## Pipeline (as reconstructed from the artifacts)

1. **Chunking** — each PDF split into ~400-character chunks with 50-character overlap.
   Each chunk keeps its source `doc_id`, `directory`, and `chunk_id` for traceability.
2. **Embedding** — chunks embedded with `all-MiniLM-L6-v2` (384-dimensional vectors).
3. **Vector store** — embeddings indexed in **ChromaDB** (cosine similarity).
4. Retrieval at query time: embed the question, pull the top-K nearest chunks, feed
   them as context to a QA/generation step (not included in this export).

## What's inside

```
data/                              <- empty; drop raw source PDFs here if you have them
models/
  chunks.pkl                         <- 76 chunk records (text + doc_id + chunk_id + char count)
  corpus_metadata.json               <- corpus stats: docs, chunk config, embedding model, dims
  embeddings.npy                     <- (76, 384) float32 embedding matrix, one row per chunk
  vector_db/
    chroma.sqlite3                    <- ChromaDB collection metadata
    <collection-uuid>/                <- ChromaDB's HNSW index files (data_level0.bin, etc.)
```

## Reloading / querying the vector store

```python
import chromadb

client = chromadb.PersistentClient(path="models/vector_db")
collection = client.get_or_create_collection("documind")  # confirm the actual collection name

results = collection.query(
    query_texts=["How many days a week can I work remotely?"],
    n_results=3,
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(meta, "->", doc[:120])
```

Or work directly from the raw arrays if you'd rather not go through Chroma:

```python
import numpy as np, pickle

embeddings = np.load("models/embeddings.npy")       # (76, 384)
with open("models/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)                          # list of 76 dicts: doc_id, chunk_id, text, chars
```

## Next phase

[Phase 6](../Phase-06_TriNLP-Deployment/) is the deployment phase — a FastAPI backend
with a Streamlit UI serving sentiment/NER/QA over HTTP.
