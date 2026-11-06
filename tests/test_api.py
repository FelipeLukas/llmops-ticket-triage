import json

from fastapi.testclient import TestClient

from app import llm
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_triage_billing_pt():
    r = client.post("/triage", json={"text": "Fui cobrado duas vezes na fatura, quero reembolso"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "billing"
    assert body["language"] == "pt"
    assert "needs_human_review" in body


def test_urgent_goes_to_human_review():
    r = client.post("/triage", json={"text": "Meu pedido não chegou, é urgente!"})
    assert r.status_code == 200
    assert r.json()["needs_human_review"] is True


def test_rejects_too_short_text():
    assert client.post("/triage", json={"text": "oi"}).status_code == 422


def test_retries_when_llm_returns_invalid_json():
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def complete(self, text, feedback=None):
            self.calls += 1
            if self.calls == 1:
                return "isso não é json", 10, 5
            good = {
                "category": "technical",
                "priority": "low",
                "language": "pt",
                "summary": "App travou",
                "suggested_reply": "Vamos ajudar.",
                "confidence": 0.9,
            }
            return "```json\n" + json.dumps(good) + "\n```", 10, 20

    flaky = Flaky()
    result = llm.triage(flaky, "O app travou ao abrir")
    assert flaky.calls == 2
    assert result.category == "technical"


def test_gives_up_after_max_attempts():
    class Broken:
        name = "broken"

        def complete(self, text, feedback=None):
            return "{}", 1, 1

    try:
        llm.triage(Broken(), "qualquer texto aqui")
        assert False, "deveria ter lançado LLMError"
    except llm.LLMError:
        pass


def test_metrics_endpoint_exposes_llm_counters():
    client.post("/triage", json={"text": "Não consigo fazer login na minha conta"})
    text = client.get("/metrics/").text
    assert "llm_requests_total" in text
