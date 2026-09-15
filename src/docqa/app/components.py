"""Pure presentation helpers for the Streamlit app.

No `streamlit` import here on purpose — this is unit-tested, and main.py stays a
thin declarative shell over these.
"""

from __future__ import annotations

from docqa.retrieval.types import RetrievedChunk


def score_chips(chunk: RetrievedChunk) -> list[str]:
    """One short label per retrieval stage that saw this chunk, in pipeline order."""
    chips: list[str] = []
    if chunk.dense_rank is not None:
        chips.append(f"dense #{chunk.dense_rank} · {chunk.dense_score:.2f}")
    if chunk.sparse_rank is not None:
        chips.append(f"sparse #{chunk.sparse_rank} · {chunk.sparse_score:.2f}")
    if chunk.rrf_score is not None:
        chips.append(f"rrf {chunk.rrf_score:.3f}")
    if chunk.rerank_score is not None:
        chips.append(f"rerank {chunk.rerank_score:.2f}")
    return chips


def source_status(index: int, n_in_context: int, used: list[int]) -> tuple[str | None, str]:
    """For the `index`-th retrieved chunk (0-based), return (citation_label, status).

    Chunks 0..n_in_context-1 made it into the prompt as [1]..[n]; the rest were
    retrieved but dropped by the context token budget.
    """
    if index < n_in_context:
        n = index + 1
        if n in used:
            return f"[{n}]", "cited in the answer"
        return f"[{n}]", "in context, not cited"
    return None, "retrieved, dropped by token budget"


def truncate(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
