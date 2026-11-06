
# llmops-ticket-triage
=======
# ticket-triage-llm

API de triagem de tickets de suporte com LLM, empacotada com Docker e instrumentada como um serviço de produção: validação de saída, retry, métricas de custo/latência, CI e scan de imagem.

O foco não é "chamar um LLM", e sim **o que acontece em volta dele** (confiabilidade, observabilidade e segurança), que é onde a maior parte do trabalho de GenAI em produção acontece.

## O que ele faz

`POST /triage` recebe o texto de um ticket (PT ou EN) e devolve:

```json
{
  "category": "billing",
  "priority": "medium",
  "language": "pt",
  "summary": "Cliente cobrado duas vezes na fatura",
  "suggested_reply": "Obrigado pelo contato, já estamos verificando.",
  "confidence": 0.85,
  "needs_human_review": false
}
```

## Arquitetura

```
cliente ──► FastAPI (/triage) ──► LLM (Claude ou mock)
                │                      │
                │   valida com Pydantic; se inválido, 1 retry com o erro como feedback
                ▼
          /metrics ──► Prometheus ──► Grafana
```

| Camada | Tecnologia |
|---|---|
| API | FastAPI + Pydantic |
| LLM | Anthropic API (`LLM_PROVIDER=anthropic`) ou mock local |
| Container | Docker (usuário não-root, healthcheck) |
| Observabilidade | Prometheus + Grafana |
| CI/CD | GitHub Actions: pytest → docker build → Trivy |

## Como rodar

```bash
cp .env.example .env          # por padrão usa o mock, sem custo
docker compose up --build

curl -X POST localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"text": "Fui cobrado duas vezes na fatura, quero reembolso"}'
```

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / valor de `GRAFANA_PASSWORD`)

Para usar o modelo real, edite o `.env`: `LLM_PROVIDER=anthropic` e `ANTHROPIC_API_KEY=...`.

Testes locais:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Métricas expostas

| Métrica | Para que serve |
|---|---|
| `llm_latency_seconds` | p95 de latência por provider |
| `llm_tokens_total{direction}` | base para estimar custo |
| `llm_requests_total{status}` | taxa de erro (`ok`, `invalid`, `error`) |
| `llm_validation_failures_total` | quantas vezes o modelo quebrou o contrato |
| `triage_human_review_total` | quantos tickets foram para revisão humana |

## Avaliação (eval)

Em vez de assumir que o LLM classifica bem, o repositório mede. `evals/tickets.jsonl` tem 50 tickets (PT e EN, 10 por categoria) com categoria, prioridade e idioma rotulados à mão, e 5 casos marcados como `ambiguous` (podem ter mais de uma resposta razoável).

```bash
# LLM real: compare versões de prompt e modelos (rode da raiz do repositório)
python -m evals.run_eval --provider anthropic --prompt v1 --out evals/results/llm-v1.json
python -m evals.run_eval --provider anthropic --prompt v2 --out evals/results/llm-v2.json

# Baseline clássico de ML no mesmo conjunto (5-fold)
pip install -r requirements-eval.txt
python -m evals.baseline --out evals/results/baseline.json
```

O script reporta acurácia (com e sem os casos ambíguos), precisão/recall/F1 por categoria, confusões mais comuns, latência média e p95, tokens e custo estimado (`--price-in` / `--price-out`).

- `v1`: instrução e contrato de saída.
- `v2`: `v1` + definição de cada categoria, critério de prioridade e 3 exemplos (few-shot) que não estão no conjunto de avaliação.
- Baseline: TF-IDF + regressão logística. É a régua: o LLM só se justifica se superá-la o bastante para compensar custo e latência.

### Resultados

Preencha após rodar com a chave de API (o provider `mock` só valida o script, não mede nenhum modelo):

| Abordagem | Acurácia (categoria) | F1 pior classe | Latência p95 | Custo / 1000 tickets |
|---|---|---|---|---|
| TF-IDF + regressão logística | | | | |
| LLM, prompt v1 | | | | |
| LLM, prompt v2 | | | | |

Limitações honestas: 50 tickets é uma validação pequena, não um benchmark; os dados são sintéticos; e os rótulos refletem o critério de uma pessoa só. Os números servem para comparar abordagens entre si, não para prometer desempenho em produção.

## Decisões de projeto

- **Saída estruturada validada, não confiada.** O LLM pode devolver JSON inválido ou fora do schema. O Pydantic valida e, se falhar, o erro é devolvido ao modelo em um retry. Se falhar de novo, a API responde 502 em vez de passar lixo adiante.
- **Humano no loop.** Tickets com baixa confiança ou prioridade alta são marcados com `needs_human_review`. A IA sugere, a pessoa confirma o que é sensível.
- **Texto do cliente tratado como dado.** O prompt delimita o ticket com tags e instrui o modelo a ignorar instruções dentro dele (mitigação básica de prompt injection).
- **Provider plugável + mock.** Dá para rodar CI e demo sem chave de API e sem custo; trocar de provider é implementar um método.
- **Imagem enxuta e segura.** `python:slim`, usuário não-root e Trivy no pipeline bloqueando CVEs críticas/altas com correção disponível.

## Limitações e próximos passos

- [ ] Ampliar o conjunto de avaliação com tickets reais anonimizados e mais de um anotador
- [ ] Medir concordância entre anotadores nos casos ambíguos
- [ ] Dashboard do Grafana provisionado via arquivo (hoje é criado manualmente)
- [ ] Cache de respostas para tickets repetidos
- [ ] Rate limiting e autenticação na API
- [ ] Deploy em Kubernetes com HPA, reaproveitando a estrutura do meu projeto `k8s-network-monitor`
