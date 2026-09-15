"""Pure Streamlit render helpers."""

from __future__ import annotations

from docqa.app.components import score_chips, source_status, truncate
from docqa.retrieval.types import RetrievedChunk


def _chunk(**kw) -> RetrievedChunk:
    base = dict(id="c", url="u", title="t", section_path=["A", "B"], text="x", token_count=1)
    base.update(kw)
    return RetrievedChunk(**base)


def test_score_chips_only_includes_stages_that_saw_the_chunk():
    c = _chunk(dense_rank=3, dense_score=0.71, rrf_score=0.032, rerank_score=2.15)
    chips = score_chips(c)
    assert chips == ["dense #3 · 0.71", "rrf 0.032", "rerank 2.15"]      # no sparse chip
    assert score_chips(_chunk()) == []


def test_score_chips_keeps_pipeline_order():
    c = _chunk(sparse_rank=1, sparse_score=0.4, dense_rank=2, dense_score=0.6)
    assert score_chips(c)[0].startswith("dense")
    assert score_chips(c)[1].startswith("sparse")


def test_source_status_transitions():
    # 3 chunks in context (n=1..3), answer cited [1] and [3]
    assert source_status(0, 3, [1, 3]) == ("[1]", "cited in the answer")
    assert source_status(1, 3, [1, 3]) == ("[2]", "in context, not cited")
    assert source_status(4, 3, [1, 3]) == (None, "retrieved, dropped by token budget")


def test_truncate():
    assert truncate("short", 100) == "short"
    assert truncate("x" * 50, 10) == "xxxxxxxxxx…"
    assert truncate("  padded  ", 100) == "padded"
