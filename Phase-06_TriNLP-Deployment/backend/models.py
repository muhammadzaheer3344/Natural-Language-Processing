"""
TriNLP Model Loading
Loads all three models: sentiment, NER, and QA.
"""

import os
from pathlib import Path

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


class ModelManager:
    """Manages all three NLP models."""
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.qa_model = None
        self.qa_tokenizer = None
    
    def load_all(self):
        """Load all three models."""
        self.load_sentiment()
        self.load_ner()
        self.load_qa()
    
    def load_sentiment(self):
        """Load sentiment analysis model."""
        model_name = self._model_path(
            "sentiment", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device
        )
        print("Sentiment model loaded")
    
    def load_ner(self):
        """Load NER model."""
        model_name = self._model_path("ner", "dslim/bert-base-NER")
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="max",
            device=self.device
        )
        print("NER model loaded")
    
    def load_qa(self):
        """Load QA model."""
        qa_name = self._model_path("qa", "deepset/bert-base-cased-squad2")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_name)
        self.qa_model = self.qa_model.to(self.device_str)
        self.qa_model.eval()
        print("QA model loaded")

    @staticmethod
    def _model_path(task, hub_name):
        model_root = os.environ.get("TRINLP_MODEL_DIR")
        if model_root:
            local_path = Path(model_root) / task
            if local_path.is_dir():
                return str(local_path)
        return hub_name
    
    def predict_sentiment(self, text):
        """Predict sentiment."""
        result = self.sentiment_pipeline(text)[0]
        return {"label": result["label"], "score": result["score"]}
    
    def predict_ner(self, text):
        """Extract named entities."""
        raw = self.ner_pipeline(text)
        return [
            {
                "text": e["word"],
                "type": e["entity_group"],
                "start": e["start"],
                "end": e["end"],
                "score": e["score"]
            }
            for e in raw
        ]
    
    def predict_qa(self, question, context):
        """Answer question given context."""
        inputs = self.qa_tokenizer(
            question, context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        ).to(self.qa_model.device)
        sequence_ids = inputs.sequence_ids(0)
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        context_tokens = [
            index for index, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and offset_mapping[index][1] > offset_mapping[index][0]
        ]
        if not context_tokens:
            return {"answer": "", "score": 0.0}

        max_answer_length = 30
        candidates = []
        for start_idx in context_tokens:
            for end_idx in context_tokens:
                if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_length:
                    continue
                candidates.append((start_idx, end_idx))

        candidate_scores = torch.stack([
            start_logits[0, start_idx] + end_logits[0, end_idx]
            for start_idx, end_idx in candidates
        ])
        best_index = torch.argmax(candidate_scores).item()
        start_idx, end_idx = candidates[best_index]

        context_start_logits = start_logits[0, context_tokens]
        context_end_logits = end_logits[0, context_tokens]
        start_probabilities = torch.softmax(context_start_logits, dim=0)
        end_probabilities = torch.softmax(context_end_logits, dim=0)
        start_position = context_tokens.index(start_idx)
        end_position = context_tokens.index(end_idx)
        score = torch.sqrt(
            start_probabilities[start_position] * end_probabilities[end_position]
        ).item()

        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return {
            "answer": answer,
            "score": score
        }


# Singleton instance
model_manager = ModelManager()
