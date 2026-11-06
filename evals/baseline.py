"""Baseline clássico de ML: TF-IDF + regressão logística, no mesmo conjunto de tickets.

Serve de régua para o LLM: o modelo grande só se justifica se superar isto o
bastante para compensar custo e latência. Como o conjunto é pequeno, usa
validação cruzada estratificada (5 folds) em vez de um único split.

    pip install -r requirements-eval.txt
    python -m evals.baseline --out evals/results/baseline.json
"""

import argparse
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .run_eval import CATEGORIES, load_dataset


def build_model() -> object:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_dataset()
    texts = [r["text"] for r in rows]
    gold = [r["category"] for r in rows]

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    start = time.perf_counter()
    pred = cross_val_predict(build_model(), texts, gold, cv=cv)
    elapsed_ms = (time.perf_counter() - start) * 1000

    report = classification_report(gold, pred, labels=CATEGORIES, output_dict=True, zero_division=0)
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)

    non_amb = [i for i, r in enumerate(rows) if not r.get("ambiguous")]
    acc_non_amb = sum(gold[i] == pred[i] for i in non_amb) / len(non_amb)

    result = {
        "config": {"model": "tfidf+logreg", "folds": args.folds, "seed": args.seed},
        "n": len(rows),
        "category_accuracy": round(accuracy, 3),
        "category_accuracy_excluding_ambiguous": round(acc_non_amb, 3),
        "per_label": {k: {m: round(v, 3) for m, v in report[k].items()} for k in CATEGORIES},
        "avg_ms_per_ticket_train_plus_predict": round(elapsed_ms / len(rows), 2),
    }

    print(f"baseline TF-IDF + regressão logística ({args.folds}-fold, seed {args.seed})")
    print(f"acurácia categoria: {accuracy:.1%} (sem os ambíguos: {acc_non_amb:.1%})")
    for label in CATEGORIES:
        m = report[label]
        print(f"  {label:<10} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1-score']:.2f} (n={int(m['support'])})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresultado salvo em {out}")


if __name__ == "__main__":
    main()
