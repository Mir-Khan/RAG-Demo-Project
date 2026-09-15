"""Ask the multi-agent pipeline a question from the CLI.

    python -m docqa.agents.run "How do I add CORS middleware?" --corpus fastapi
    python -m docqa.agents.run "..." --provider fake      # offline smoke test

Shows the routing decision, the sources used, the cited answer, and per-node timings.
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query")
    p.add_argument("--corpus", default=None)
    p.add_argument("--provider", default=None, help="override LLM_PROVIDER (groq|ollama|fake)")
    args = p.parse_args(argv)

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        from docqa.config import get_settings

        get_settings.cache_clear()

    from docqa.agents.graph import QAPipeline

    pipe = QAPipeline(corpus=args.corpus)
    console.rule(f"[bold]{args.query}")
    result = pipe.answer(args.query)

    tag = "[yellow]fallback[/]" if result.fallback_used else "[green]routed[/]"
    console.print(f"{tag} → [bold]{result.category}[/]  (confidence {result.confidence:.2f})")

    console.print("\n[bold]sources[/]")
    for c in result.citations:
        mark = "[green]•[/]" if c.n in result.used_citation_numbers else "[dim]◦[/]"
        console.print(f"  {mark} [{c.n}] [dim]{c.breadcrumb or c.title}[/]  {c.url}")

    console.print(Panel(result.answer, title="answer", border_style="cyan"))
    console.print(f"[dim]timings: {result.timings_ms}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
