"""End-to-end ingestion: config -> loader -> chunk -> embed -> Postgres.

Run:  python -m docqa.ingestion.pipeline [--fresh] [--limit N] [--dry-run]

Incremental by default: for every source doc touched this run we delete its old
chunks then insert the new ones, so a doc that now produces fewer chunks doesn't
leave stale rows. `--fresh` wipes the whole corpus first.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from docqa.config import get_settings, load_corpus_config
from docqa.ingestion.chunker import chunk_document, default_token_counter
from docqa.ingestion.loaders import build_loader

console = Console()


def run(fresh: bool = False, limit: int | None = None, dry_run: bool = False) -> None:
    settings = get_settings()
    corpus = load_corpus_config(settings.corpus)
    console.rule(f"[bold]Ingest: {corpus.display_name or corpus.name}")

    loaders = [build_loader(lc) for lc in corpus.loaders]
    counter = default_token_counter(settings.embedding_model)
    console.print(f"token counter: [cyan]{type(counter).__name__}[/]")

    docs = []
    for lc, loader in zip(corpus.loaders, loaders, strict=True):
        batch = list(loader.load())
        console.print(f"  [dim]{lc.type}[/]: {len(batch)} docs")
        docs.extend(batch)
    if limit:
        docs = docs[:limit]
    console.print(f"loaded [green]{len(docs)}[/] source docs")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, corpus.name, corpus.chunking, counter))

    tok = [c.token_count for c in all_chunks] or [0]
    console.print(
        f"produced [green]{len(all_chunks)}[/] chunks  "
        f"(tokens: min {min(tok)}, avg {sum(tok) // len(tok)}, max {max(tok)})"
    )

    if dry_run:
        for c in all_chunks[:5]:
            console.print(f"\n[dim]{' > '.join(c.section_path)}[/]  [yellow]{c.token_count}t[/]")
            console.print(c.text[:300] + ("…" if len(c.text) > 300 else ""))
        console.print("\n[dim]--dry-run: nothing written[/]")
        return

    # DB work is imported lazily so --dry-run needs no psycopg / running Postgres.
    import numpy as np

    from docqa.db import store
    from docqa.ingestion.embed import Embedder

    embedder = Embedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        expected_dim=settings.embedding_dim,
    )
    console.print(f"embedding [cyan]{len(all_chunks)}[/] chunks with {embedder.model_name}…")
    vectors = embedder.embed_passages([c.embed_text or c.text for c in all_chunks])

    conn = store.connect(settings.database_url)
    try:
        store.init_schema(conn)
        if fresh:
            n = store.delete_corpus(conn, corpus.name)
            console.print(f"[yellow]--fresh[/]: deleted {n} existing rows")
        else:
            touched = sorted({c.source_id for c in all_chunks})
            store.delete_sources(conn, corpus.name, touched)
        store.upsert_chunks(conn, all_chunks, np.asarray(vectors))
        console.print("[green]upsert complete[/]")
        console.print(store.corpus_stats(conn, corpus.name))
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest a documentation corpus into pgvector.")
    p.add_argument("--fresh", action="store_true", help="delete the whole corpus before ingesting")
    p.add_argument("--limit", type=int, help="only process the first N source docs")
    p.add_argument("--dry-run", action="store_true", help="chunk only; no embedding, no DB")
    args = p.parse_args(argv)
    run(fresh=args.fresh, limit=args.limit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
