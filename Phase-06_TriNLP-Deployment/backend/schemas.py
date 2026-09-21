from pydantic import BaseModel
from typing import List

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

class NERRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    score: float

class NERResponse(BaseModel):
    entities: List[Entity]

class QARequest(BaseModel):
    question: str
    context: str

class QAResponse(BaseModel):
    answer: str
    score: float
