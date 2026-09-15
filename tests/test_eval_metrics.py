"""Deterministic metric scorers."""

from __future__ import annotations

from docqa.eval.metrics import (
    aggregate,
    is_refusal,
    reference_hit_rank,
    retrieval_recall_at_k,
    router_correct,
)

A = "https://fastapi.tiangolo.com/tutorial/cors/"
B = "https://fastapi.tiangolo.com/tutorial/handling-errors/"
C = "https://fastapi.tiangolo.com/tutorial/body/"


def test_recall_at_k_ignores_trailing_slash_and_fragment():
    retrieved = [A + "#setup", "https://fastapi.tiangolo.com/tutorial/cors"]
    assert retrieval_recall_at_k(retrieved, [A]) == 1.0


def test_recall_at_k_partial_and_none():
    assert retrieval_recall_at_k([A], [A, B]) == 0.5
    assert retrieval_recall_at_k([A], []) is None          # negative item


def test_reference_hit_rank_is_first_matching_position():
    retrieved = ["https://x/other/", B, A]
    assert reference_hit_rank(retrieved, [A]) == 3
    assert reference_hit_rank(retrieved, ["https://x/none/"]) is None


def test_is_refusal_detects_common_phrasings():
    assert is_refusal("The provided sources do not cover rate limiting.")
    assert is_refusal("I could not find information about that in the documentation.")
    # present-tense "does not cover" / contracted "doesn't cover" — actual model output,
    # missed by an earlier cue list that only had "not covered" (regression caught in
    # eval-v-iter1b: real refusals scored as non-refusals because of this substring gap)
    assert is_refusal("The documentation does not cover per-API-key rate limiting in FastAPI.")
    assert is_refusal("Sorry, this doesn't cover that topic.")
    assert not is_refusal("Use CORSMiddleware with allow_origins set to your frontend URL [1].")


def test_router_correct():
    assert router_correct("conceptual", "conceptual")
    assert not router_correct("conceptual", "fallback")


def _row(**kw):
    base = dict(
        difficulty="single_chunk", recall_at_k=1.0, router_correct=True,
        fallback_used=False, is_refusal=False,
    )
    base.update(kw)
    return base


def test_aggregate_shapes_and_values():
    rows = [
        _row(recall_at_k=1.0),
        _row(recall_at_k=0.0, router_correct=False),
        _row(difficulty="multi_chunk", recall_at_k=0.5),
        _row(difficulty="negative", recall_at_k=None, is_refusal=True),
        _row(difficulty="negative", recall_at_k=None, is_refusal=False, fallback_used=True),
    ]
    agg = aggregate(rows)
    assert agg["n"] == 5
    # recall over non-negative rows: mean(1.0, 0.0, 0.5)
    assert agg["retrieval_recall_at_k"] == 0.5
    assert agg["router_accuracy"] == round(4 / 5, 4)
    assert agg["negative_refusal_rate"] == 0.5
    assert agg["fallback_rate"] == 0.2
    assert "single_chunk" in agg["by_difficulty"]
