# Phase 1 — SMS Shield: Classical ML

SMS spam detection using TF-IDF features and classical machine-learning classifiers.
This is the baseline phase: no embeddings, no deep learning — just proven, fast, and
highly interpretable classical ML.

## Dataset

- Source: [Kaggle SMS Spam Collection Dataset](https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv) (loaded directly by URL in the notebook)
- ~5,500 SMS messages labeled `ham` (legitimate) / `spam`
- Class-imbalanced (spam is the minority class)

## Approach

1. Text cleaning: lowercasing, punctuation removal, stopword removal (**negations kept**
   — "not", "no", "never" etc. are preserved since they flip meaning), lemmatization.
2. Feature extraction: **TF-IDF vectorization**.
3. Three classical models trained and compared on the validation set:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (`LinearSVC`)
4. Best model (SVM) re-evaluated on the held-out test set.

## Result — SVM, final test set

| Metric | Score |
|---|---|
| Accuracy | 0.9858 |
| Precision | 1.0000 |
| Recall | 0.8878 |
| F1 | 0.9405 |

Precision of 1.00 means **zero false positives** — no legitimate message was ever
flagged as spam, at the cost of a small number of missed spam messages (recall 0.89).
For a spam filter, that's the right trade-off: a missed spam message is annoying, a
blocked real message is worse.

## What's inside

```
SMS_Shield_Classical_ML.ipynb   <- full notebook: EDA, cleaning, TF-IDF, 3 models, evaluation
models/
  sms_shield_tfidf_vectorizer.pkl   <- fitted TF-IDF vectorizer (needed to transform new text)
  sms_shield_svm_model.pkl          <- trained LinearSVC model (the final classifier)
```

## Reloading the model

```python
import pickle

with open("models/sms_shield_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/sms_shield_svm_model.pkl", "rb") as f:
    model = pickle.load(f)

text = ["Congratulations! You've won a free prize, call now!"]
X = vectorizer.transform(text)
prediction = model.predict(X)   # 'spam' or 'ham'
```

## Next phase

[Phase 2](../Phase-02_SMS-Shield-Neural-Network/) tackles the exact same problem with a
learned-embedding neural network instead of TF-IDF + classical ML.
