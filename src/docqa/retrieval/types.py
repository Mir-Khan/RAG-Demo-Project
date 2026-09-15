"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    id: object                         # uuid.UUID from Postgres
    url: str
    title: str
    section_path: list[str]
    text: str
    token_count: int

    # per-stage scores/ranks — None means "this stage didn't see this chunk"
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_rank: int | None = None

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    @property
    def preview(self) -> str:
        one_line = " ".join(self.text.split())
        return one_line[:120] + ("…" if len(one_line) > 120 else "")


@dataclass
class RetrievalTrace:
    """Everything the retriever produced for one query — the final answer plus the
    intermediate rankings, so a change to any stage is inspectable."""

    query: str
    corpus: str
    results: list[RetrievedChunk]           # final, trimmed to final_k
    dense: list[RetrievedChunk] = field(default_factory=list)
    sparse: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranked: bool = False
