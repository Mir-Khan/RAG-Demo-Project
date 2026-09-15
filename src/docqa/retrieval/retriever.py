"""Hybrid retrieval: dense + sparse -> RRF -> cross-encoder rerank.

    Retriever(corpus="fastapi").retrieve("How do I add a CORS middleware?")
        -> list[RetrievedChunk]  (final_k chunks, best first)

Models and the DB connection are created once per Retriever and reused, so the
agents and the Streamlit app should hold a single long-lived instance.
"""

from __future__ import annotations

import time

import psycopg

from docqa.config import Settings, get_settings, load_corpus_config
from docqa.db import store
from docqa.ingestion.embed import Embedder
from docqa.retrieval.fusion import reciprocal_rank_fusion
from docqa.retrieval.rerank import Reranker
from docqa.retrieval.types import RetrievedChunk, RetrievalTrace


def _hit_to_chunk(h: dict) -> RetrievedChunk:
    return RetrievedChunk(
        id=h["id"],
        url=h["url"],
        title=h["title"],
        section_path=list(h["section_path"] or []),
        text=h["text"],
        token_count=h["token_count"],
    )


class Retriever:
    def __init__(
        self,
        corpus: str | None = None,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus_cfg = load_corpus_config(corpus or self.settings.corpus)
        self.rcfg = self.corpus_cfg.retrieval
        self._embedder = embedder
        self._reranker = reranker
        self._conn = None

    # -- lazily-built, reused resources ---------------------------------- #
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                self.settings.embedding_model,
                query_prefix=self.settings.embedding_query_prefix,
                expected_dim=self.settings.embedding_dim,
            )
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.settings.reranker_model)
        return self._reranker

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = store.connect(self.settings.database_url)
        return self._conn

    def _db(self, fn):
        """Run a store.* call, reconnecting once if the connection was dropped.
        Managed Postgres (Neon free tier) suspends compute on idle and kills open
        connections with AdminShutdown; the next connect wakes it (~1s)."""
        try:
            return fn(self.conn)
        except (psycopg.OperationalError, psycopg.InterfaceError):
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._conn = None
            return fn(self.conn)

    # -- public API ----------------------------------------------------- #
    def retrieve(
        self, query: str, *, rerank: bool | None = None, final_k: int | None = None
    ) -> list[RetrievedChunk]:
        return self.search(query, rerank=rerank, final_k=final_k).results

    def search(
        self, query: str, *, rerank: bool | None = None, final_k: int | None = None
    ) -> RetrievalTrace:
        rcfg = self.rcfg
        do_rerank = rcfg.rerank if rerank is None else rerank
        final_k = final_k or rcfg.final_k
        corpus = self.corpus_cfg.name
        timings: dict[str, float] = {}

        # 1. dense (vector) ------------------------------------------------
        t = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        dense_hits = self._db(lambda c: store.vector_search(c, corpus, qvec, rcfg.dense_k))
        timings["dense_ms"] = (time.perf_counter() - t) * 1000

        # 2. sparse (full-text) -----------------------------------------
        t = time.perf_counter()
        sparse_hits = self._db(lambda c: store.keyword_search(c, corpus, query, rcfg.sparse_k))
        timings["sparse_ms"] = (time.perf_counter() - t) * 1000

        # one RetrievedChunk per id, annotated by whichever stages saw it
        by_id: dict[object, RetrievedChunk] = {}
        dense_ids: list[object] = []
        for rank, h in enumerate(dense_hits, 1):
            rc = by_id.setdefault(h["id"], _hit_to_chunk(h))
            rc.dense_rank, rc.dense_score = rank, float(h["score"])
            dense_ids.append(h["id"])
        sparse_ids: list[object] = []
        for rank, h in enumerate(sparse_hits, 1):
            rc = by_id.setdefault(h["id"], _hit_to_chunk(h))
            rc.sparse_rank, rc.sparse_score = rank, float(h["score"])
            sparse_ids.append(h["id"])

        # 3. Reciprocal Rank Fusion -----------------------------------
        fused_scores = reciprocal_rank_fusion([dense_ids, sparse_ids], k=rcfg.rrf_k)
        for cid, s in fused_scores.items():
            by_id[cid].rrf_score = s
        fused = sorted(by_id.values(), key=lambda c: c.rrf_score or 0.0, reverse=True)
        for rank, rc in enumerate(fused, 1):
            rc.rrf_rank = rank

        # 4. cross-encoder rerank -----------------------------------
        results = fused
        if do_rerank and fused:
            cand = fused[: rcfg.rerank_top_n]
            t = time.perf_counter()
            scores = self.reranker.score(query, [c.text for c in cand])
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000
            for rc, sc in zip(cand, scores, strict=True):
                rc.rerank_score = sc
            cand.sort(key=lambda c: c.rerank_score, reverse=True)
            results = cand
        for rank, rc in enumerate(results, 1):
            rc.final_rank = rank

        return RetrievalTrace(
            query=query,
            corpus=corpus,
            results=results[:final_k],
            dense=[by_id[i] for i in dense_ids],
            sparse=[by_id[i] for i in sparse_ids],
            fused=fused,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            reranked=bool(do_rerank and fused),
        )
