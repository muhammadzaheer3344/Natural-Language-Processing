# Phase 4 — CineMood: BERT

Same IMDB movie review sentiment problem as Phase 3, this time fine-tuning a pretrained
transformer — **DistilBERT** — instead of training a sequence model from scratch.

## Dataset

- IMDB movie reviews, loaded via Hugging Face `datasets` (`stanfordnlp/imdb`)
- Same positive/negative sentiment task as Phase 3, so results are directly comparable

## Approach

1. Tokenized with the `distilbert-base-uncased` tokenizer (max length 256).
2. `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`
   — a pretrained DistilBERT with a classification head on top.
3. Fine-tuned end-to-end on the IMDB training set (GPU).
4. Evaluated on the held-out test set.

## Result — final test set

| Metric | Score |
|---|---|
| Test Loss | 0.2744 |
| Accuracy | **0.9148** |
| Precision | 0.9123 |
| Recall | 0.9179 |
| F1 | **0.9151** |

Confusion matrix (25,000 test reviews): 11,397 negative correctly identified, 1,103
negative misclassified as positive, 1,026 positive misclassified as negative, 11,474
positive correctly identified.

## Comparison with Phase 3 (same problem, different technique)

| Model | Accuracy | F1 |
|---|---|---|
| GRU (Phase 3) | 0.8689 | 0.8668 |
| **DistilBERT (Phase 4)** | **0.9148** | **0.9151** |

Fine-tuning a pretrained transformer beats the best from-scratch recurrent model by
**~4.6 accuracy points** — the value of transfer learning from a model that already
"knows" English before it ever sees an IMDB review.

## What's inside

```
CineMood_BERT.ipynb              <- notebook: tokenization, fine-tuning loop, evaluation
models/
  cinemood_bert/
    config.json                    <- model architecture/config
    model.safetensors               <- fine-tuned model weights
    tokenizer.json                  <- fast tokenizer
    tokenizer_config.json           <- tokenizer settings
```

## Reloading the model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("models/cinemood_bert")
model = AutoModelForSequenceClassification.from_pretrained("models/cinemood_bert")
model.eval()

inputs = tokenizer("This movie was absolutely wonderful.", return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits
prediction = torch.softmax(logits, dim=1)   # [prob_negative, prob_positive]
```

## Next phase

[Phase 5](../Phase-05_DocuMind-RAG/) moves past classification entirely, into
retrieval-augmented generation for document question-answering.
