"""
models.py — Pydantic request / response models for the SHL agent API.

Schema is non-negotiable per the evaluator spec; do not rename fields.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Literal


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool