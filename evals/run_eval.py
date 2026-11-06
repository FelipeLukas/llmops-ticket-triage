"""Avalia o pipeline de triagem contra evals/tickets.jsonl.

Exemplos (rode da raiz do repositório):
    python -m evals.run_eval --provider anthropic --prompt v1 --out evals/results/haiku-v1.json
    python -m evals.run_eval --provider anthropic --prompt v2 --out evals/results/haiku-v2.json
    python -m evals.run_eval --provider anthropic --model <outro-modelo> --prompt v2

Com --provider mock o script só valida o encanamento: o mock é heurística de
palavras-chave e o resultado NÃO mede a qualidade de nenhum modelo.
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from app import llm

DATA = Path(__file__).parent / "tickets.jsonl"
CATEGORIES = ["billing", "technical", "account", "delivery", "other"]
PRIORITIES = ["low", "medium", "high"]
LANGUAGES = ["pt", "en"]


def load_dataset(path: Path = DATA) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def compute_metrics(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    """Acurácia geral e precisão/recall/F1 por classe. pred=None (falha) conta como erro."""
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    per_label = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }
    return {"accuracy": round(correct / n, 3) if n else 0.0, "n": n, "per_label": per_label}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


class Metered:
    """Envolve um cliente LLM para somar chamadas e tokens durante o eval."""

    def __init__(self, inner: llm.LLMClient) -> None:
        self.inner = inner
        self.name = inner.name
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def complete(self, text: str, feedback: str | None = None) -> tuple[str, int, int]:
        raw, tok_in, tok_out = self.inner.complete(text, feedback)
        self.calls += 1
        self.tokens_in += tok_in
        self.tokens_out += tok_out
        return raw, tok_in, tok_out


def run(client: Metered, rows: list[dict]) -> dict:
    preds, latencies, failures, errors = [], [], 0, []
    for row in rows:
        start = time.perf_counter()
        try:
            result = llm.triage(client, row["text"])
            pred = {"category": result.category, "priority": result.priority, "language": result.language}
        except llm.LLMError:
            pred = {"category": None, "priority": None, "language": None}
            failures += 1
        latencies.append(time.perf_counter() - start)
        preds.append(pred)
        if pred["category"] != row["category"]:
            errors.append(
                {"id": row["id"], "text": row["text"], "gold": row["category"],
                 "pred": pred["category"], "ambiguous": row.get("ambiguous", False)}
            )

    def col(field: str) -> list:
        return [r[field] for r in rows]

    def pcol(field: str) -> list:
        return [p[field] for p in preds]

    non_ambiguous = [i for i, r in enumerate(rows) if not r.get("ambiguous")]
    return {
        "n": len(rows),
        "failures": failures,
        "category": compute_metrics(col("category"), pcol("category"), CATEGORIES),
        "category_excluding_ambiguous": compute_metrics(
            [rows[i]["category"] for i in non_ambiguous],
            [preds[i]["category"] for i in non_ambiguous],
            CATEGORIES,
        )["accuracy"],
        "priority": compute_metrics(col("priority"), pcol("priority"), PRIORITIES),
        "language": compute_metrics(col("language"), pcol("language"), LANGUAGES),
        "latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 1),
            "p95": round(percentile(latencies, 95) * 1000, 1),
        },
        "tokens": {"input": client.tokens_in, "output": client.tokens_out, "calls": client.calls},
        "category_errors": errors,
        "confusions": Counter(f"{e['gold']}->{e['pred']}" for e in errors).most_common(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=None, help="mock | anthropic (padrão: LLM_PROVIDER)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="v1", choices=sorted(llm.PROMPTS))
    parser.add_argument("--limit", type=int, default=None, help="usar só os N primeiros tickets")
    parser.add_argument("--out", default=None, help="salvar o resultado completo em JSON")
    parser.add_argument("--price-in", type=float, default=None, help="USD por 1M tokens de entrada")
    parser.add_argument("--price-out", type=float, default=None, help="USD por 1M tokens de saída")
    args = parser.parse_args()

    rows = load_dataset()[: args.limit]
    client = Metered(llm.get_client(args.provider, args.model, args.prompt))
    if client.name == "mock":
        print("ATENÇÃO: provider mock. Isto só valida o script, não mede nenhum modelo.\n")

    report = run(client, rows)
    report["config"] = {
        "provider": client.name,
        "model": getattr(client.inner, "model", None),
        "prompt": args.prompt,
    }
    if args.price_in is not None and args.price_out is not None:
        cost = (client.tokens_in * args.price_in + client.tokens_out * args.price_out) / 1_000_000
        report["estimated_cost_usd"] = round(cost, 4)

    cat = report["category"]
    print(f"config: {report['config']}")
    print(f"tickets: {report['n']} | falhas de formato: {report['failures']}")
    print(f"acurácia categoria: {cat['accuracy']:.1%} "
          f"(sem os ambíguos: {report['category_excluding_ambiguous']:.1%})")
    print(f"acurácia prioridade: {report['priority']['accuracy']:.1%}")
    print(f"acurácia idioma: {report['language']['accuracy']:.1%}")
    print(f"latência: média {report['latency_ms']['mean']} ms | p95 {report['latency_ms']['p95']} ms")
    print(f"tokens: {report['tokens']}")
    if "estimated_cost_usd" in report:
        print(f"custo estimado: US$ {report['estimated_cost_usd']}")
    print("\nF1 por categoria:")
    for label, m in cat["per_label"].items():
        print(f"  {label:<10} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")
    if report["confusions"]:
        print("\nconfusões (gabarito->previsto):", report["confusions"])

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresultado salvo em {out}")


if __name__ == "__main__":
    main()
