from typing import Literal

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    text: str = Field(min_length=5, max_length=4000, description="Texto do ticket do cliente")


class Triage(BaseModel):
    """Contrato de saída que o LLM é obrigado a respeitar."""

    category: Literal["billing", "technical", "account", "delivery", "other"]
    priority: Literal["low", "medium", "high"]
    language: Literal["pt", "en", "other"]
    summary: str = Field(min_length=1, max_length=300)
    suggested_reply: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)


class TriageResponse(Triage):
    # Mesma ideia do "confirmar antes de gravar": a decisão final pode ser humana.
    needs_human_review: bool
