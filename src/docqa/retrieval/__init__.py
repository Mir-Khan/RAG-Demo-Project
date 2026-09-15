"""Hybrid retrieval package.

`fusion` and `types` are pure (stdlib only) and imported eagerly. `Retriever`
pulls in psycopg + sentence-transformers, so it's loaded lazily — unit tests of
the fusion math must not require a DB driver or the ML stack.
"""

from docqa.retrieval.fusion import reciprocal_rank_fusion
from docqa.retrieval.types import RetrievedChunk, RetrievalTrace

__all__ = ["Retriever", "RetrievedChunk", "RetrievalTrace", "reciprocal_rank_fusion"]


def __getattr__(name: str):  # PEP 562
    if name == "Retriever":
        from docqa.retrieval.retriever import Retriever

        return Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
