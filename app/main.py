import os

from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app

from . import metrics
from .llm import LLMError, get_client, triage
from .schemas import TriageRequest, TriageResponse

app = FastAPI(title="ticket-triage-llm", version="0.1.0")
app.mount("/metrics", make_asgi_app())

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
_client = get_client()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": _client.name}


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(req: TriageRequest) -> TriageResponse:
    try:
        result = triage(_client, req.text)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    needs_review = result.confidence < CONFIDENCE_THRESHOLD or result.priority == "high"
    if needs_review:
        metrics.HUMAN_REVIEW.inc()
    return TriageResponse(**result.model_dump(), needs_human_review=needs_review)
