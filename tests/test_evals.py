from collections import Counter

from evals.run_eval import CATEGORIES, LANGUAGES, PRIORITIES, compute_metrics, load_dataset, percentile


def test_dataset_is_well_formed():
    rows = load_dataset()
    assert len(rows) == 50
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["category"] in CATEGORIES
        assert r["priority"] in PRIORITIES
        assert r["language"] in LANGUAGES
        assert len(r["text"]) >= 5


def test_dataset_is_balanced_across_categories():
    counts = Counter(r["category"] for r in load_dataset())
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) == max(counts.values()) == 10


def test_dataset_has_both_languages():
    counts = Counter(r["language"] for r in load_dataset())
    assert counts["pt"] > 0 and counts["en"] > 0


def test_compute_metrics_basic():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", None]
    m = compute_metrics(gold, pred, ["a", "b"])
    assert m["accuracy"] == 0.5
    assert m["per_label"]["a"]["precision"] == 1.0
    assert m["per_label"]["a"]["recall"] == 0.5
    assert m["per_label"]["b"]["support"] == 2


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 95) == 5
    assert percentile([], 95) == 0.0
