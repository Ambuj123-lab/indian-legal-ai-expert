"""
Pydantic Schemas for RAG interactions
"""
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    use_streaming: bool = False

class FeedbackRequest(BaseModel):
    question: str
    response: str
    rating: str  # "up" or "down"
