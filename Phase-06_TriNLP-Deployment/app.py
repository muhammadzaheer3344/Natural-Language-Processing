import streamlit as st
import os

from models import model_manager


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

st.set_page_config(page_title="TriNLP", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Loading NLP models...")
def get_model_manager():
    model_manager.load_all()
    return model_manager


st.title("TriNLP")
st.caption("Sentiment analysis, named entity recognition, and question answering")

try:
    models = get_model_manager()
except Exception as exc:
    st.error("The NLP models could not be loaded.")
    st.exception(exc)
    st.stop()

tab_sentiment, tab_ner, tab_qa = st.tabs(
    ["Sentiment", "Named Entities", "Question Answering"]
)

with tab_sentiment:
    text = st.text_area(
        "Text",
        placeholder="Type or paste a sentence...",
        key="sentiment_text",
    )
    if st.button("Analyze sentiment", key="sentiment_btn", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            result = models.predict_sentiment(text.strip())
            st.metric("Prediction", result["label"], f"{result['score']:.2%} confidence")

with tab_ner:
    ner_text = st.text_area(
        "Text",
        placeholder="e.g. Elon Musk founded SpaceX in California.",
        key="ner_text",
    )
    if st.button("Extract entities", key="ner_btn", type="primary"):
        if not ner_text.strip():
            st.warning("Please enter some text.")
        else:
            entities = models.predict_ner(ner_text.strip())
            if not entities:
                st.info("No entities found.")
            else:
                st.dataframe(entities, use_container_width=True, hide_index=True)

with tab_qa:
    context = st.text_area(
        "Context",
        placeholder="Paste a paragraph the question will be answered from...",
        key="qa_context",
    )
    question = st.text_input("Question", key="qa_question")
    if st.button("Get answer", key="qa_btn", type="primary"):
        if not context.strip() or not question.strip():
            st.warning("Please fill in both the context and the question.")
        else:
            result = models.predict_qa(question.strip(), context.strip())
            st.success(result["answer"] or "No answer found.")
            st.caption(f"Confidence: {result['score']:.2%}")
