"""Compare two eval result files: aggregate deltas + per-question regressions.

    python -m docqa.eval.diff evals/results/baseline-v1.json evals/results/latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# a drop larger than this on a per-question judged metric is flagged as a regression
REGRESSION_DELTA = 0.15


def _flat_metrics(result: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for section in ("deterministic", "ragas"):
        for k, v in (result.get(section) or {}).items():
            if isinstance(v, (int, float)):
                out[k] = float(v)
    return out


def _rows_by_id(result: dict) -> dict[str, dict]:
    return {r["id"]: r for r in result.get("rows", [])}


def regressions(
    baseline: dict, current: dict, delta: float = REGRESSION_DELTA
) -> list[tuple[str, str, float, float]]:
    """Per-question (id, metric, baseline_score, current_score) where a judged
    metric dropped by more than `delta`."""
    br, cr = _rows_by_id(baseline), _rows_by_id(current)
    out: list[tuple[str, str, float, float]] = []
    for qid in sorted(br.keys() & cr.keys()):
        for m in ("faithfulness", "context_precision"):
            bv, cv = br[qid].get(m), cr[qid].get(m)
            if bv is not None and cv is not None and (bv - cv) > delta:
                out.append((qid, m, bv, cv))
    return out


def diff(baseline: dict, current: dict) -> None:
    b, c = _flat_metrics(baseline), _flat_metrics(current)
    br, cr = _rows_by_id(baseline), _rows_by_id(current)
    t = Table(title="aggregate metrics", title_style="bold")
    t.add_column("metric")
    t.add_column("baseline", justify="right")
    t.add_column("current", justify="right")
    t.add_column("Δ", justify="right")
    for key in sorted(set(b) | set(c)):
        bv, cv = b.get(key), c.get(key)
        if bv is None or cv is None:
            t.add_row(key, f"{bv}", f"{cv}", "—")
            continue
        d = cv - bv
        colour = "green" if d >= 0 else "red"
        # fallback_rate is "lower is better"; flip its colour
        if "fallback" in key:
            colour = "red" if d > 0 else "green"
        t.add_row(key, f"{bv:.4f}", f"{cv:.4f}", f"[{colour}]{d:+.4f}[/]")
    console.print(t)

    # per-question judged regressions
    flagged = regressions(baseline, current)
    if flagged:
        console.print(f"\n[bold red]per-question regressions[/] (drop > {REGRESSION_DELTA}):")
        for qid, m, bv, cv in flagged:
            q = cr.get(qid, {}).get("question", "")
            console.print(f"  [yellow]{qid}[/] {m}: {bv:.2f} → {cv:.2f}   [dim]{q[:70]}[/]")
    else:
        console.print("\n[green]no per-question faithfulness/context_precision regressions[/]")

    # routing flips
    flips = [
        (qid, br[qid].get("category_actual"), cr[qid].get("category_actual"))
        for qid in sorted(br.keys() & cr.keys())
        if br[qid].get("category_actual") != cr[qid].get("category_actual")
    ]
    if flips:
        console.print("\n[bold]routing changed:[/]")
        for qid, was, now in flips:
            console.print(f"  {qid}: {was} → {now}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    args = p.parse_args(argv)
    diff(json.loads(args.baseline.read_text()), json.loads(args.current.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
