"""Turn retrieved chunks into the numbered source block the LLM sees.

Greedy fill to a token budget: chunks are already rerank-ordered, so we add the
best first and stop when the budget is hit. Each chunk gets a [n] label the model
is told to cite; the returned Citation list maps [n] back to a real URL.
"""

from __future__ import annotations

from docqa.agents.state import Citation
from docqa.ingestion.chunker import TokenCounter
from docqa.retrieval.types import RetrievedChunk

_HEADER = "Use these numbered sources. Cite claims with the matching [n].\n"


def assemble_context(
    chunks: list[RetrievedChunk], max_tokens: int, counter: TokenCounter
) -> tuple[str, list[Citation]]:
    blocks: list[str] = []
    citations: list[Citation] = []
    used = counter.count(_HEADER)

    for chunk in chunks:
        n = len(citations) + 1
        loc = chunk.breadcrumb or chunk.title or "source"
        block = f"[{n}] {loc}\n{chunk.text}\n({chunk.url})"
        cost = counter.count(block)
        if citations and used + cost > max_tokens:
            break
        blocks.append(block)
        citations.append(
            Citation(
                n=n,
                title=chunk.title,
                url=chunk.url,
                breadcrumb=chunk.breadcrumb,
                chunk_id=str(chunk.id),
            )
        )
        used += cost

    return _HEADER + "\n" + "\n\n".join(blocks), citations
