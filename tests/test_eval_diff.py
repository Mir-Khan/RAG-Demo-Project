"""Result diffing: aggregate flattening + per-question regression detection."""

from __future__ import annotations

from docqa.eval.diff import _flat_metrics, regressions


def _result(agg_det, agg_ragas, rows):
    return {"deterministic": agg_det, "ragas": agg_ragas, "rows": rows}


def test_flat_metrics_merges_deterministic_and_ragas_numbers_only():
    r = _result(
        {"router_accuracy": 0.8, "by_difficulty": {"single_chunk": {"n": 3}}},
        {"faithfulness": 0.9},
        [],
    )
    flat = _flat_metrics(r)
    assert flat == {"router_accuracy": 0.8, "faithfulness": 0.9}   # nested dict dropped


def test_regressions_flags_only_big_drops():
    base = _result({}, {}, [
        {"id": "q1", "faithfulness": 0.90, "context_precision": 0.80},
        {"id": "q2", "faithfulness": 0.70, "context_precision": 0.60},
    ])
    cur = _result({}, {}, [
        {"id": "q1", "faithfulness": 0.60, "context_precision": 0.78},  # faithfulness -0.30 -> flag
        {"id": "q2", "faithfulness": 0.68, "context_precision": 0.59},  # tiny drops -> no flag
    ])
    flags = regressions(base, cur, delta=0.15)
    assert [(qid, m) for qid, m, _, _ in flags] == [("q1", "faithfulness")]


def test_regressions_ignores_missing_scores():
    base = _result({}, {}, [{"id": "q1"}])
    cur = _result({}, {}, [{"id": "q1", "faithfulness": 0.2}])
    assert regressions(base, cur) == []
