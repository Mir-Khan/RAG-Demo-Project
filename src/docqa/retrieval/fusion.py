"""Reciprocal Rank Fusion.

Given several ranked lists of the same items, combine them into one ranking using
ONLY each item's position, not its raw score:

    score(d) = Σ_lists  1 / (k + rank_of_d_in_list)      (rank is 1-based)

Why not normalize-and-add the real scores: cosine similarity (0..1, and usually
bunched up near the top) and ts_rank (unbounded, length-dependent) live on
different scales whose relationship shifts per query, so any weighting is brittle.
RRF sidesteps that entirely. `k` damps the influence of low-ranked items; 60 is
the value from the original RRF paper and the de-facto standard.

Pure function, no I/O — unit-tested in tests/test_fusion.py.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60
) -> dict[Hashable, float]:
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores
