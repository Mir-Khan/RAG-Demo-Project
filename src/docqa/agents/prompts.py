"""Prompt templates. Provider-neutral on purpose (see docqa.llm)."""

from __future__ import annotations

from docqa.config import CorpusConfig, RouterCategory

ROUTER_SYSTEM = """You classify a user question about {display_name} into exactly one category.

Categories:
{category_block}

Respond with a JSON object only: {{"category": "<id>", "confidence": <0.0-1.0>}}
- "category" must be one of the ids above, or "fallback" if none fit.
- "confidence" is how sure you are of the category.
Do not answer the question. Classify it."""

GEN_SYSTEM = """You are a documentation assistant for {display_name}.

Rules:
- Answer ONLY from the numbered sources. Never add facts from outside knowledge.
- If the sources describe a *related* feature but not the specific thing asked,
  say the specific thing is not in the documentation. Do NOT answer about the
  related feature as though it were what was asked.
- If the sources do not address the question at all, reply in one sentence that
  the documentation does not cover it, and stop. Do not speculate.
- Cite every factual claim with the matching [n]. Prefer the sources' own wording.
- Be concise.
{style}"""


def render_router_system(cfg: CorpusConfig) -> str:
    lines = []
    for c in cfg.router_categories:
        ex = f"  e.g. {c.examples[0]}" if c.examples else ""
        lines.append(f'- {c.id}: {c.description}{ex}')
    return ROUTER_SYSTEM.format(
        display_name=cfg.display_name or cfg.name,
        category_block="\n".join(lines),
    )


def render_gen_system(cfg: CorpusConfig, category: RouterCategory) -> str:
    style = category.answer_style.strip()
    return GEN_SYSTEM.format(
        display_name=cfg.display_name or cfg.name,
        style=f"\nFor this question type: {style}" if style else "",
    )


def render_user_turn(query: str, context: str) -> str:
    return f"{context}\n\n---\nQuestion: {query}\n\nAnswer (with [n] citations):"
