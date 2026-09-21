# Phase 2 — SMS Shield: Neural Network

Same SMS spam detection problem as Phase 1, this time with a Keras neural network
using learned word embeddings instead of TF-IDF — the first step from classical ML
towards deep learning.

## Dataset

Same as Phase 1: [Kaggle SMS Spam Collection Dataset](https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv), same cleaning
pipeline (lowercasing, negation-preserving stopword removal, lemmatization).

## Approach

1. Text tokenized with Keras `Tokenizer`, sequences padded to a fixed length.
2. Model architecture:
   - `Embedding` layer (learned word vectors, not TF-IDF)
   - `GlobalAveragePooling1D`
   - `Dense(64, activation='relu')`
   - `Dropout`
   - `Dense(1, activation='sigmoid')` output
3. Trained with early stopping on validation loss.

## Result — final test set

| Metric | Score |
|---|---|
| Accuracy | 0.9806 |
| Precision | 0.9560 |
| Recall | 0.8878 |
| F1 | 0.9206 |

Confusion matrix (test set, 775 messages): 673 ham correctly kept, 4 ham misflagged as
spam, 11 spam missed, 87 spam correctly caught.

## Comparison with Phase 1

| Metric | Phase 1 (SVM + TF-IDF) | Phase 2 (NN + embeddings) |
|---|---|---|
| Accuracy | **0.9858** | 0.9806 |
| Precision | **1.0000** | 0.9560 |
| Recall | 0.8878 | 0.8878 |
| F1 | **0.9405** | 0.9206 |

On this dataset size (~5,500 rows), the classical TF-IDF + SVM approach from Phase 1
actually edges out the neural network — a useful, honest lesson: more complex ≠
automatically better, especially on a small, relatively clean dataset.

## What's inside

```
SMS_Shield_Neural_Network.ipynb   <- notebook: tokenization, model, training curves, evaluation
models/
  sms_shield_tokenizer.pkl          <- fitted Keras Tokenizer (pickled)
  sms_shield_nn_model.keras         <- trained Keras model
```

## Reloading the model

```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("models/sms_shield_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("models/sms_shield_nn_model.keras")

seq = tokenizer.texts_to_sequences(["Congratulations! You've won a free prize!"])
padded = pad_sequences(seq, maxlen=100)  # match MAX_LENGTH used in training
prob = model.predict(padded)[0][0]        # probability of spam
```

## Next phase

[Phase 3](../Phase-03_CineMood-LSTM-GRU/) moves to a different problem (movie review
sentiment) and a sequence-aware architecture (LSTM/GRU) that actually reads word order.
