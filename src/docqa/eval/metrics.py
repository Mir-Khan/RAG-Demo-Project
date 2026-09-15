"""Deterministic metrics — no LLM, no API key, no network.

These grade the parts of the system that don't need a judge: did retrieval fetch
the right pages, did the router classify correctly, and did the system refuse when
the answer genuinely isn't in the corpus.
"""

from __future__ import annotations

import statistics
from urllib.parse import urldefrag

_REFUSAL_CUES = (
    "not cover", "n't cover",              # "not covered" / "does not cover" / "doesn't cover"
    "not address", "n't address",
    "not in the", "not mention", "n't mention",
    "do not have", "don't have", "no information",
    "sources do not", "sources don't",
    "cannot find", "could not find", "n't find",
    "not provided in", "not present in", "not appear in",
    "outside the scope", "not documented",
)


def norm_url(url: str) -> str:
    return urldefrag(url or "")[0].rstrip("/").lower()


def retrieval_recall_at_k(retrieved_urls: list[str], reference_urls: list[str]) -> float | None:
    """Fraction of the reference (should-retrieve) pages that appear anywhere in
    the retrieved set. None when the item has no references (negatives)."""
    if not reference_urls:
        return None
    want = {norm_url(u) for u in reference_urls}
    got = {norm_url(u) for u in retrieved_urls}
    return len(want & got) / len(want)


def reference_hit_rank(retrieved_urls: list[str], reference_urls: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk whose page is a reference page."""
    want = {norm_url(u) for u in reference_urls}
    for i, u in enumerate(retrieved_urls, 1):
        if norm_url(u) in want:
            return i
    return None


def is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(cue in a for cue in _REFUSAL_CUES)


def router_correct(expected: str, actual: str) -> bool:
    return expected == actual


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def aggregate(rows: list[dict]) -> dict:
    """rows: dicts produced by eval.run, each with the deterministic fields filled."""
    non_neg = [r for r in rows if r["difficulty"] != "negative"]
    neg = [r for r in rows if r["difficulty"] == "negative"]

    by_difficulty: dict[str, dict] = {}
    for d in ("single_chunk", "multi_chunk"):
        sub = [r for r in rows if r["difficulty"] == d]
        if sub:
            by_difficulty[d] = {
                "n": len(sub),
                "recall_at_k": _mean([r["recall_at_k"] for r in sub]),
            }

    return {
        "n": len(rows),
        "retrieval_recall_at_k": _mean([r["recall_at_k"] for r in non_neg]),
        "reference_hit_rate": _mean(
            [1.0 if r["recall_at_k"] and r["recall_at_k"] > 0 else 0.0 for r in non_neg]
        ),
        "router_accuracy": _mean([1.0 if r["router_correct"] else 0.0 for r in rows]),
        "fallback_rate": _mean([1.0 if r["fallback_used"] else 0.0 for r in rows]),
        "negative_refusal_rate": _mean([1.0 if r["is_refusal"] else 0.0 for r in neg]),
        "by_difficulty": by_difficulty,
    }
