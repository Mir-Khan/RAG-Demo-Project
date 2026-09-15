-- Single-table design: vectors live next to the text, metadata and full-text
-- index they belong with, so hybrid retrieval + corpus scoping is one SQL query
-- with real transactions (no dual-write sync problem). See docs/architecture.md §3.3.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           uuid PRIMARY KEY,                       -- uuid5(corpus:source_id:ordinal)
    corpus       text        NOT NULL,
    source_id    text        NOT NULL,
    url          text        NOT NULL DEFAULT '',
    title        text        NOT NULL DEFAULT '',
    section_path text[]      NOT NULL DEFAULT '{}',
    ordinal      integer     NOT NULL,
    text         text        NOT NULL,
    token_count  integer     NOT NULL DEFAULT 0,
    embedding    vector(384),                            -- bge-small-en-v1.5, L2-normalised
    -- Full-text vector over title + breadcrumb + body, so a keyword hit on a heading
    -- term still ranks the chunk. Maintained by upsert_chunks(), not a GENERATED
    -- column: the text-config form of to_tsvector() is STABLE, not IMMUTABLE, so
    -- Postgres rejects it in a generation expression.
    tsv          tsvector,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (corpus, source_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING gin (tsv);
-- HNSW: incremental inserts, good recall/latency at this scale, only ef_search to
-- tune at query time. IVFFlat would need rebuilds as data drifts. See §3.4.
CREATE INDEX IF NOT EXISTS chunks_embed_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
