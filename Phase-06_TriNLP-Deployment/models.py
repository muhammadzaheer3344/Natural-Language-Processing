"""Model loading and inference for the Streamlit application."""

import os
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class ModelManager:
    """Load and run the sentiment, NER, and question-answering models."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.qa_model = None
        self.qa_tokenizer = None

    def load_all(self):
        self.load_sentiment()
        self.load_ner()
        self.load_qa()

    def load_sentiment(self):
        model_name = self._model_path(
            "sentiment", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", model=model_name, device=self.device
        )

    def load_ner(self):
        model_name = self._model_path("ner", "dslim/bert-base-NER")
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="max",
            device=self.device,
        )

    def load_qa(self):
        model_name = self._model_path("qa", "deepset/bert-base-cased-squad2")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_model = self.qa_model.to(self.device_str)
        self.qa_model.eval()

    @staticmethod
    def _model_path(task, hub_name):
        model_root = os.environ.get("TRINLP_MODEL_DIR")
        if model_root:
            local_path = Path(model_root) / task
            if local_path.is_dir():
                return str(local_path)
        return hub_name

    def predict_sentiment(self, text):
        result = self.sentiment_pipeline(text)[0]
        return {"label": result["label"], "score": result["score"]}

    def predict_ner(self, text):
        return [
            {
                "text": entity["word"],
                "type": entity["entity_group"],
                "start": entity["start"],
                "end": entity["end"],
                "score": entity["score"],
            }
            for entity in self.ner_pipeline(text)
        ]

    def predict_qa(self, question, context):
        inputs = self.qa_tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        ).to(self.qa_model.device)
        sequence_ids = inputs.sequence_ids(0)
        offset_mapping = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        context_tokens = [
            index
            for index, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and offset_mapping[index][1] > offset_mapping[index][0]
        ]
        if not context_tokens:
            return {"answer": "", "score": 0.0}

        candidates = [
            (start_index, end_index)
            for start_index in context_tokens
            for end_index in context_tokens
            if end_index >= start_index and end_index - start_index + 1 <= 30
        ]
        candidate_scores = torch.stack(
            [
                outputs.start_logits[0, start_index]
                + outputs.end_logits[0, end_index]
                for start_index, end_index in candidates
            ]
        )
        start_index, end_index = candidates[torch.argmax(candidate_scores).item()]

        context_start_logits = outputs.start_logits[0, context_tokens]
        context_end_logits = outputs.end_logits[0, context_tokens]
        start_position = context_tokens.index(start_index)
        end_position = context_tokens.index(end_index)
        score = torch.sqrt(
            torch.softmax(context_start_logits, dim=0)[start_position]
            * torch.softmax(context_end_logits, dim=0)[end_position]
        ).item()

        answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
        answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return {"answer": answer, "score": score}


model_manager = ModelManager()