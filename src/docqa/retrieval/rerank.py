"""Cross-encoder reranker.

Retrieval (dense + sparse) is a BI-ENCODER stage: query and chunk are turned into
vectors *independently*, so the model never compares them directly — fast but
approximate. A CROSS-ENCODER takes (query, chunk) as one input and outputs a
single relevance score, so it can model word-level interaction. Far more precise
at "does this passage actually answer the question", but it's one model forward
pass per candidate, so we only run it on the ~20 survivors of fusion.

bge-reranker-base emits an unbounded logit (higher = more relevant); we keep it
raw for ranking and expose a sigmoid only for human-readable debug output.
"""

from __future__ import annotations

import math


class Reranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        return [float(s) for s in self._model.predict(pairs)]

    @staticmethod
    def to_probability(logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit))
