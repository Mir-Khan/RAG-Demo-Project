"""Run the evaluation harness.

    python -m docqa.eval.run --split dev                    # tune against this
    python -m docqa.eval.run --split holdout --label holdout-after-tuning
    python -m docqa.eval.run --label baseline-v1            # -> evals/results/baseline-v1.json
    python -m docqa.eval.run --no-ragas --limit 5           # fast, deterministic only

For every eval item it runs the full QAPipeline (route -> retrieve -> sub-agent),
scores the deterministic metrics always and the RAGAS metrics unless --no-ragas,
writes a timestamped JSON under evals/results/, and diffs it against the newest
committed baseline-*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from docqa.config import REPO_ROOT, get_settings, load_corpus_config
from docqa.eval import metrics as det
from docqa.eval.dataset import DEFAULT_PATH, load_eval_set

console = Console()
RESULTS_DIR = REPO_ROOT / "evals" / "results"


def _config_snapshot(settings, corpus_cfg) -> dict:
    return {
        "corpus": corpus_cfg.name,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.llm_model,
        "eval_judge_model": settings.eval_judge_model,
        "chunking": corpus_cfg.chunking.model_dump(),
        "retrieval": corpus_cfg.retrieval.model_dump(),
        "router_min_confidence": settings.router_min_confidence,
        "max_context_tokens": settings.max_context_tokens,
    }


def _latest_baseline() -> Path | None:
    files = sorted(RESULTS_DIR.glob("baseline*.json"))
    return files[-1] if files else None


def run(
    split: str,
    label: str | None,
    use_ragas: bool,
    limit: int | None,
    provider: str | None = None,
    judge_model: str | None = None,
) -> Path:
    # A tuning loop makes dozens of LLM calls; point it at a local model to escape
    # hosted free-tier RPM/RPD caps.  --provider ollama  covers both the pipeline
    # generator and the RAGAS judge.
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if judge_model:
        os.environ["EVAL_JUDGE_MODEL"] = judge_model
    if provider or judge_model:
        get_settings.cache_clear()

    settings = get_settings()
    corpus_cfg = load_corpus_config(settings.corpus)
    items = load_eval_set(DEFAULT_PATH, split=split)
    if limit:
        items = items[:limit]
    console.rule(f"[bold]eval: {corpus_cfg.name}  split={split}  n={len(items)}")

    from docqa.agents.graph import QAPipeline

    pipe = QAPipeline(corpus=corpus_cfg.name, settings=settings)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, item in enumerate(items, 1):
        res = pipe.answer(item.question)
        retrieved_urls = [c.url for c in res.trace.results]
        contexts = [c.text for c in res.trace.results]
        recall = det.retrieval_recall_at_k(retrieved_urls, item.reference_urls)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "difficulty": item.difficulty,
                "split": item.split,
                "category_expected": item.expected_category,
                "category_actual": res.category,
                "router_correct": det.router_correct(item.expected_category, res.category),
                "fallback_used": res.fallback_used,
                "confidence": round(res.confidence, 3),
                "answer": res.answer,
                "reference_urls": item.reference_urls,
                "retrieved_urls": retrieved_urls,
                "contexts": contexts,
                "recall_at_k": recall,
                "reference_hit_rank": det.reference_hit_rank(retrieved_urls, item.reference_urls),
                "is_refusal": det.is_refusal(res.answer),
                "timings_ms": res.timings_ms,
            }
        )
        console.print(
            f"  [{i}/{len(items)}] {item.id}  route={res.category}"
            f"{'*' if res.fallback_used else ''}  recall@k={recall}"
        )
    wall = round(time.perf_counter() - t0, 1)

    deterministic = det.aggregate(rows)
    ragas_agg: dict = {}
    if use_ragas:
        from docqa.eval import ragas_metrics

        if not ragas_metrics.available():
            console.print("[yellow]ragas / langchain extras not installed — skipping judged metrics[/]")
        else:
            console.print(
                f"[cyan]scoring with RAGAS (judge = {settings.llm_provider}:"
                f"{settings.eval_judge_model or 'default'})…[/]"
            )
            try:
                ragas_agg, per_row = ragas_metrics.score_with_ragas(
                    rows,
                    provider=settings.llm_provider,
                    judge_model=settings.eval_judge_model,
                    groq_api_key=settings.groq_api_key,
                    google_api_key=settings.google_api_key,
                    ollama_base_url=settings.ollama_base_url,
                    embedding_model=settings.embedding_model,
                )
                for row, scores in zip(rows, per_row, strict=True):
                    row.update(scores)
                # faithfulness and context_precision are both ill-defined on
                # negatives: a correct "not in the docs" refusal has no supporting
                # context, and there is no relevant chunk to retrieve. Report the
                # non-negative subset so the headline isn't dragged by the system
                # doing the right thing. See docs/architecture.md §3.10.
                def _mean_non_negative(metric: str) -> float | None:
                    xs = [
                        r[metric]
                        for r in rows
                        if r["difficulty"] != "negative" and r.get(metric) is not None
                    ]
                    return round(sum(xs) / len(xs), 4) if xs else None

                for m in ("faithfulness", "context_precision"):
                    val = _mean_non_negative(m)
                    if val is not None:
                        ragas_agg[f"{m}_excl_negatives"] = val
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]RAGAS failed:[/] {exc}\n[dim](deterministic metrics still saved)[/]")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "n_items": len(items),
        "wall_seconds": wall,
        "config": _config_snapshot(settings, corpus_cfg),
        "deterministic": deterministic,
        "ragas": ragas_agg,
        "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{label}.json" if label else f"{stamp}.json"
    out = RESULTS_DIR / name
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    console.rule("[bold]summary")
    console.print({"deterministic": deterministic, "ragas": ragas_agg})
    console.print(f"[green]written[/] {out.relative_to(REPO_ROOT)}   [dim]({wall}s wall)[/]")

    base = _latest_baseline()
    if base and base != out:
        console.rule(f"[bold]diff vs {base.name}")
        from docqa.eval.diff import diff

        diff(json.loads(base.read_text()), result)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    p.add_argument("--label", default=None, help="output filename stem (e.g. baseline-v1)")
    p.add_argument("--no-ragas", action="store_true", help="deterministic metrics only")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--provider", default=None,
                   help="override LLM_PROVIDER for this run (ollama recommended — no rate limits)")
    p.add_argument("--judge-model", default=None, help="override EVAL_JUDGE_MODEL")
    args = p.parse_args(argv)
    run(args.split, args.label, use_ragas=not args.no_ragas, limit=args.limit,
        provider=args.provider, judge_model=args.judge_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
