"""Context assembly: numbering, token budget, citation mapping."""

from __future__ import annotations

from docqa.agents.context import assemble_context
from docqa.ingestion.chunker import HeuristicTokenCounter
from docqa.retrieval.types import RetrievedChunk

COUNTER = HeuristicTokenCounter()


def _chunk(i: int, words: int = 20) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"id{i}",
        url=f"https://docs/{i}",
        title=f"Page {i}",
        section_path=["Tutorial", f"Section {i}"],
        text=" ".join(["word"] * words),
        token_count=words,
    )


def test_sources_are_numbered_from_one():
    ctx, cites = assemble_context([_chunk(1), _chunk(2), _chunk(3)], 10_000, COUNTER)
    assert [c.n for c in cites] == [1, 2, 3]
    assert "[1]" in ctx and "[2]" in ctx and "[3]" in ctx
    assert "https://docs/2" in ctx


def test_token_budget_truncates_but_keeps_at_least_one():
    chunks = [_chunk(i, words=200) for i in range(10)]
    ctx, cites = assemble_context(chunks, max_tokens=120, counter=COUNTER)
    assert 1 <= len(cites) < 10          # stopped early
    assert cites[0].n == 1


def test_citation_maps_back_to_url_and_breadcrumb():
    _, cites = assemble_context([_chunk(7)], 10_000, COUNTER)
    assert cites[0].url == "https://docs/7"
    assert cites[0].breadcrumb == "Tutorial > Section 7"
    assert cites[0].chunk_id == "id7"


def test_empty_input():
    ctx, cites = assemble_context([], 1000, COUNTER)
    assert cites == []
    assert isinstance(ctx, str)
