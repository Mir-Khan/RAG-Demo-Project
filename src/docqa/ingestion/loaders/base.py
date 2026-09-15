"""The ingestion seam.

Everything corpus-specific stops here. A loader's only job is to turn some source
(a GitHub repo, a folder of markdown, a crawl) into a stream of `RawDoc`s with a
consistent shape. The chunker, embedder, store, retrieval, agents and UI all
operate on that shape and never learn what the corpus actually is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class RawDoc:
    source_id: str                       # stable id within the corpus, e.g. "tutorial/query-params"
    title: str                           # human title, usually the page's H1
    url: str                             # canonical public URL for citations
    body_markdown: str                   # cleaned markdown, ready for the chunker
    section_path: list[str] = field(default_factory=list)  # breadcrumb prefix, e.g. ["Tutorial"]


@runtime_checkable
class SourceLoader(Protocol):
    def load(self) -> Iterable[RawDoc]: ...
