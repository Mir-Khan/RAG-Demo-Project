"""Graph state + the public result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from docqa.retrieval.types import RetrievalTrace


class QAState(TypedDict, total=False):
    """LangGraph working state. Each node writes a disjoint set of keys, so no
    channel reducers are needed."""

    query: str
    # written by `route`
    category: str
    confidence: float
    fallback_used: bool
    # written by `retrieve`
    trace: RetrievalTrace
    context: str
    citations: list["Citation"]
    # written by the chosen sub-agent
    answer: str
    # timings, one key per node
    t_route_ms: float
    t_retrieve_ms: float
    t_generate_ms: float


@dataclass(slots=True)
class Citation:
    n: int
    title: str
    url: str
    breadcrumb: str
    chunk_id: str


@dataclass
class AnswerResult:
    query: str
    corpus: str
    category: str
    confidence: float
    fallback_used: bool
    answer: str
    citations: list[Citation]
    used_citation_numbers: list[int]
    trace: RetrievalTrace
    timings_ms: dict[str, float] = field(default_factory=dict)
