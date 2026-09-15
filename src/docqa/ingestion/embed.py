"""Embedding wrapper around sentence-transformers.

bge-small-en-v1.5 specifics we bake in:
  * L2-normalise every vector, so cosine distance == (1 - dot) and pgvector's
    `vector_cosine_ops` index behaves predictably.
  * Apply the training-time instruction prefix to QUERIES ONLY. Passages get none.
    Prefixing passages too (a common bug) measurably hurts retrieval.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "", expected_dim: int | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix.strip()
        self._model = SentenceTransformer(model_name)
        try:
            self.dim = self._model.get_embedding_dimension()          # sentence-transformers >= 3.3
        except AttributeError:
            self.dim = self._model.get_sentence_embedding_dimension()  # older
        if expected_dim is not None and self.dim != expected_dim:
            raise ValueError(
                f"{model_name} produces {self.dim}-dim vectors but config/schema expect {expected_dim}. "
                "Changing embedding model is a schema migration (see docs/architecture.md appendix)."
            )

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        q = f"{self.query_prefix} {text}".strip() if self.query_prefix else text
        return self._model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
