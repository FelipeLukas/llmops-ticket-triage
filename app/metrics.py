from prometheus_client import Counter, Histogram

LLM_REQUESTS = Counter(
    "llm_requests_total", "Chamadas ao LLM", ["provider", "status"]
)
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "Latência das chamadas ao LLM",
    ["provider"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
LLM_TOKENS = Counter(
    "llm_tokens_total", "Tokens consumidos", ["provider", "direction"]
)
LLM_VALIDATION_FAILURES = Counter(
    "llm_validation_failures_total",
    "Respostas do LLM que não passaram na validação do schema",
    ["provider"],
)
HUMAN_REVIEW = Counter(
    "triage_human_review_total", "Tickets encaminhados para revisão humana"
)
