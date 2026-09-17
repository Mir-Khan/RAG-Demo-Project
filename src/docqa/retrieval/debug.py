"""Inspect the retrieval pipeline stage by stage.

    python -m docqa.retrieval.debug "how do I handle a 422 error" --corpus fastapi --k 10
    python -m docqa.retrieval.debug "..." --no-rerank        # see the pre-rerank order

Prints four tables (dense / sparse / fused / final) plus a short "what moved"
analysis. This is the tool for understanding *why* retrieval returns what it does.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from docqa.retrieval.rerank import Reranker
from docqa.retrieval.retriever import Retriever
from docqa.retrieval.types import RetrievedChunk

console = Console()


def _table(title: str, rows: list[RetrievedChunk], score_of, k: int) -> Table:
    t = Table(title=title, title_style="bold cyan", show_lines=False, expand=True)
    t.add_column("#", justify="right", width=3)
    t.add_column("score", justify="right", width=8)
    t.add_column("breadcrumb", style="dim", max_width=40, no_wrap=True)
    t.add_column("preview")
    for i, rc in enumerate(rows[:k], 1):
        s = score_of(rc)
        t.add_row(str(i), f"{s:.4f}" if s is not None else "—", rc.breadcrumb, rc.preview)
    return t


def _short_id(rc: RetrievedChunk) -> str:
    return str(rc.id)[:8]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query")
    p.add_argument("--corpus", default=None, help="corpus name (default: $CORPUS)")
    p.add_argument("--k", type=int, default=10, help="rows to show per table")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder stage")
    args = p.parse_args(argv)

    retriever = Retriever(corpus=args.corpus)
    console.rule(f"[bold]query:[/] {args.query}   [dim]corpus={retriever.corpus_cfg.name}[/]")
    trace = retriever.search(args.query, rerank=not args.no_rerank)

    console.print(_table("1 · DENSE (vector / cosine)", trace.dense, lambda r: r.dense_score, args.k))
    console.print(_table("2 · SPARSE (full-text / ts_rank)", trace.sparse, lambda r: r.sparse_score, args.k))
    console.print(_table("3 · FUSED (RRF)", trace.fused, lambda r: r.rrf_score, args.k))
    if trace.reranked:
        console.print(_table("4 · RERANKED (cross-encoder)", trace.results, lambda r: r.rerank_score, args.k))

    # --- what moved -----------------------------------------------------
    dense_set = {r.id for r in trace.dense}
    sparse_set = {r.id for r in trace.sparse}
    only_dense = [r for r in trace.dense if r.id not in sparse_set]
    only_sparse = [r for r in trace.sparse if r.id not in dense_set]
    both = [r for r in trace.fused if r.id in dense_set and r.id in sparse_set]

    console.rule("[bold]analysis")
    console.print(f"[green]{len(both)}[/] chunks found by BOTH methods "
                  f"(these dominate the fused top — agreement is a strong signal).")
    if only_dense:
        console.print(f"\n[yellow]{len(only_dense)}[/] dense-only (missed by keywords — likely paraphrase / synonym):")
        for r in only_dense[:3]:
            console.print(f"  [dim]{_short_id(r)}[/] d#{r.dense_rank:<2} {r.breadcrumb}")
    if only_sparse:
        console.print(f"\n[yellow]{len(only_sparse)}[/] sparse-only (missed by vectors — likely exact token / identifier):")
        for r in only_sparse[:3]:
            console.print(f"  [dim]{_short_id(r)}[/] s#{r.sparse_rank:<2} {r.breadcrumb}")

    if trace.reranked and trace.fused:
        pre = {r.id: i for i, r in enumerate(trace.fused, 1)}
        console.print("\n[bold]rerank movement[/] (fused rank -> final rank):")
        for i, r in enumerate(trace.results, 1):
            delta = pre.get(r.id, 0) - i
            if delta > 0:
                arrow = f"[green]+{delta}[/]"
            elif delta < 0:
                arrow = f"[red]{delta}[/]"
            else:
                arrow = "·"
            console.print(f"  {i:>2}. was #{pre.get(r.id,'?'):<3} {arrow:>10}  {r.breadcrumb}")

    console.print(f"\n[dim]timings: {trace.timings_ms}[/]")
    console.print(f"[dim]probabilities (sigmoid of rerank logit): "
                  f"{[round(Reranker.to_probability(r.rerank_score), 3) for r in trace.results if r.rerank_score is not None]}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
