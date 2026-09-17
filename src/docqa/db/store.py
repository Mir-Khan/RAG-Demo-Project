"""Thin Postgres/pgvector data-access layer. Raw SQL on purpose — the point of
choosing pgvector was to lean on SQL, not hide it behind an ORM."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from docqa.ingestion.chunker import Chunk

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(dsn: str) -> psycopg.Connection:
    # autocommit: psycopg3 defaults to autocommit=False, which means every plain
    # SELECT silently opens a transaction that's never closed unless something
    # calls commit()/rollback(). The Retriever holds one connection across many
    # read-only queries and across idle time between user questions — left in
    # non-autocommit mode, that connection sits "idle in transaction", and Neon
    # (correctly) kills sessions that do that:
    # "terminating connection due to idle-in-transaction timeout". Every statement
    # here is either a single read or an already-atomic write, so autocommit is
    # not just a workaround, it's the correct mode for this connection's actual
    # usage pattern (confirmed: an explicit .commit() elsewhere is a harmless
    # no-op under autocommit, so nothing else needs to change).
    #
    # TCP keepalives so idle connections through NATs/proxies aren't silently
    # dropped; the retriever also reconnects on OperationalError for the cases
    # keepalives can't prevent (e.g. Neon's compute-suspend).
    conn = psycopg.connect(
        dsn,
        autocommit=True,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )
    # pgvector's type adapter can only bind once the `vector` type exists, and a
    # fresh managed database won't have the extension yet. Creating it here is
    # idempotent and is a precondition for every query this app runs.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets psycopg adapt numpy arrays <-> pgvector
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(_SCHEMA_PATH.read_text(encoding="utf-8"))


def delete_corpus(conn: psycopg.Connection, corpus: str) -> int:
    cur = conn.execute("DELETE FROM chunks WHERE corpus = %s", (corpus,))
    return cur.rowcount


def delete_sources(conn: psycopg.Connection, corpus: str, source_ids: Sequence[str]) -> int:
    if not source_ids:
        return 0
    cur = conn.execute(
        "DELETE FROM chunks WHERE corpus = %s AND source_id = ANY(%s)",
        (corpus, list(source_ids)),
    )
    return cur.rowcount


# `tsv` is built from a single text param (title + breadcrumb + body, joined in
# Python) rather than SQL string-wrangling — keeps psycopg out of polymorphic-type
# inference and keeps the FTS source obvious.
_INSERT = """
INSERT INTO chunks
    (id, corpus, source_id, url, title, section_path, ordinal, text, token_count, embedding, tsv)
VALUES
    (%(id)s, %(corpus)s, %(source_id)s, %(url)s, %(title)s, %(section_path)s,
     %(ordinal)s, %(text)s, %(token_count)s, %(embedding)s,
     to_tsvector('english', %(fts_text)s))
ON CONFLICT (id) DO UPDATE SET
    url          = EXCLUDED.url,
    title        = EXCLUDED.title,
    section_path = EXCLUDED.section_path,
    ordinal      = EXCLUDED.ordinal,
    text         = EXCLUDED.text,
    token_count  = EXCLUDED.token_count,
    embedding    = EXCLUDED.embedding,
    tsv          = EXCLUDED.tsv
"""


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    rows = [
        {
            "id": c.id,
            "corpus": c.corpus,
            "source_id": c.source_id,
            "url": c.url,
            "title": c.title,
            "section_path": c.section_path,
            "ordinal": c.ordinal,
            "text": c.text,
            "token_count": c.token_count,
            "embedding": np.asarray(emb, dtype=np.float32),
            "fts_text": " ".join([c.title, *c.section_path, c.text]),
        }
        for c, emb in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(_INSERT, rows)


# --------------------------------------------------------------------------- #
# Retrieval queries (stage 2)                                                  #
# --------------------------------------------------------------------------- #
# `<=>` is pgvector's cosine DISTANCE (0 = identical, 2 = opposite). Vectors are
# L2-normalised at ingest, so `1 - distance` is a clean 0..1 similarity.
_VECTOR_SQL = """
SELECT id, url, title, section_path, text, token_count,
       1 - (embedding <=> %(qv)s::vector) AS score
FROM chunks
WHERE corpus = %(corpus)s AND embedding IS NOT NULL
ORDER BY embedding <=> %(qv)s::vector
LIMIT %(limit)s
"""

# `websearch_to_tsquery` parses Google-style input safely but ANDs every term, so a
# natural-language question ("why am I getting a 422 unprocessable entity response")
# only matches a chunk containing *all* of {422, unprocess, entiti, respons} — for
# scattered documentation that's usually nothing. Rewrite the `&`s to `|`s: a chunk
# matches on any term, and `ts_rank_cd` (cover density: term frequency + proximity)
# still ranks chunks that hit more terms, closer together, at the top. Phrases
# ("...") and negation stay intact.
_KEYWORD_SQL = """
WITH q AS (
    SELECT websearch_to_tsquery('english', %(q)s) AS and_q,
           replace(websearch_to_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS or_q
)
SELECT c.id, c.url, c.title, c.section_path, c.text, c.token_count,
       -- OR match ranks partial hits; the AND term is 0 unless the chunk contains
       -- every query token, so chunks with the full rare phrase ("422 unprocessable
       -- entity response") leapfrog chunks that just repeat a common word.
       ts_rank_cd(c.tsv, q.or_q) + 3.0 * ts_rank_cd(c.tsv, q.and_q) AS score
FROM chunks c, q
WHERE c.corpus = %(corpus)s AND c.tsv @@ q.or_q
ORDER BY score DESC
LIMIT %(limit)s
"""


def vector_search(
    conn: psycopg.Connection, corpus: str, query_vec, limit: int
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _VECTOR_SQL,
            {"qv": np.asarray(query_vec, dtype=np.float32), "corpus": corpus, "limit": limit},
        )
        return cur.fetchall()


def keyword_search(
    conn: psycopg.Connection, corpus: str, query_text: str, limit: int
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_KEYWORD_SQL, {"q": query_text, "corpus": corpus, "limit": limit})
        return cur.fetchall()


def corpus_stats(conn: psycopg.Connection, corpus: str) -> dict:
    row = conn.execute(
        """
        SELECT count(*)                       AS n_chunks,
               count(DISTINCT source_id)      AS n_sources,
               coalesce(avg(token_count), 0)  AS avg_tokens,
               coalesce(max(token_count), 0)  AS max_tokens
        FROM chunks WHERE corpus = %s
        """,
        (corpus,),
    ).fetchone()
    return {
        "n_chunks": row[0],
        "n_sources": row[1],
        "avg_tokens": round(float(row[2]), 1),
        "max_tokens": row[3],
    }
