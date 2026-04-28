from typing import Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to score")
    threshold: Optional[float] = Field(
        default=None,
        description="Decision threshold. If set, returns 'AI'/'Human' labels.",
    )


class TextScore(BaseModel):
    text: str
    score: float
    label: Optional[str] = None


class ScoreResponse(BaseModel):
    results: list[TextScore]
    model: str
    threshold: Optional[float]
