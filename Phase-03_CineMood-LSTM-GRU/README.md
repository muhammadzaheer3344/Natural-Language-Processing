# Phase 3 — CineMood: LSTM / GRU

Movie review sentiment classification (positive / negative) on the IMDB dataset, using
sequence models — LSTM, GRU, and Bidirectional LSTM — that read reviews as an ordered
sequence of words instead of a bag of features.

## Dataset

- IMDB movie reviews, loaded directly from `tensorflow.keras.datasets.imdb`
- Vocabulary capped at the top 10,000 most frequent words
- Sequences padded/truncated to a fixed length of 400 tokens

## Approach

Three architectures were built and compared, all sharing the same shape
(`Embedding → recurrent layer → Dropout → Dense(1, sigmoid)`):

1. **LSTM** (with `mask_zero=True` so padding is ignored)
2. **GRU** (same shape, GRU cell instead of LSTM)
3. **Bidirectional LSTM** (reads the sequence forwards and backwards)

## Results — final test set

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LSTM | 0.8578 | 0.8301 | 0.8996 | 0.8635 |
| **GRU** | **0.8689** | 0.8809 | 0.8531 | **0.8668** |
| BiLSTM | 0.8601 | 0.8874 | 0.8248 | 0.8550 |

**GRU was selected as the final model** — highest accuracy and F1, with a simpler
architecture (fewer parameters) than the BiLSTM. That's the model saved to `models/`.

## What's inside

```
CineMood_LSTM_GRU.ipynb   <- notebook: data loading, all 3 architectures, comparison
models/
  cinemood_gru.keras        <- trained GRU model (the best of the three)
```

## Reloading the model

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("models/cinemood_gru.keras")

# To score new text you'll need the same word_index used at training time:
word_index = imdb.get_word_index()
# map words -> integers (+3 offset, as Keras' IMDB loader does), then:
padded = pad_sequences([sequence], maxlen=400, padding='post', truncating='post')
prob = model.predict(padded)[0][0]   # probability of positive sentiment
```

## Next phase

[Phase 4](../Phase-04_CineMood-BERT/) tackles the same IMDB sentiment problem with a
fine-tuned transformer (DistilBERT) — the jump from recurrence to attention.
