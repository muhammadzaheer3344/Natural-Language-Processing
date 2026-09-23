# TriNLP

TriNLP is a Streamlit app for sentiment analysis, named entity recognition, and question answering. It loads pretrained models from Hugging Face and runs inference directly in the Streamlit process. There is no separate API server or frontend service.

## Run locally

```bash
cd Phase-06_TriNLP-Deployment
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app at https://share.streamlit.io/.
3. Select this repository and set the main file to `Phase-06_TriNLP-Deployment/app.py`.

Models download from Hugging Face the first time the app starts. Streamlit's resource cache keeps them loaded while the app is running.

## Files

```
app.py             Streamlit UI and app entrypoint
models.py          Model loading and inference logic
requirements.txt   Streamlit Cloud dependencies
```

The models are `distilbert-base-uncased-finetuned-sst-2-english`, `dslim/bert-base-NER`, and `deepset/bert-base-cased-squad2`.
