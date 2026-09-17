"""Header/section-aware chunking.

WHY NOT fixed-size splitting
---------------------------
Naive `text[i:i+N]` (or a fixed token window) cuts mid-sentence and mid-concept.
It also destroys structure: a chunk can end up containing the tail of one topic
and the head of the next, so its embedding is a blurred average of two things and
matches neither query well. Docs already carry a human-authored hierarchy —
headings. We use it.

ALGORITHM
---------
1. Parse the markdown line-by-line, tracking a heading stack (H1..H6) and skipping
   anything inside fenced code blocks (so `# not a heading` in a shell example is
   safe).
2. Emit one "section" per heading: its text is everything from that heading down
   to the next heading of any level. Each section knows its full breadcrumb, e.g.
   ["Tutorial", "Query Parameters", "Optional parameters"].
3. Merge sections smaller than `min_tokens` into a neighbour (prefer backward — a
   short "Recap" attaches to what it follows) so stubs never become chunks.
4. Emit chunks:
     * section fits in `max_tokens`  -> one chunk
     * section too big               -> split on paragraph boundaries into
                                        overlapping windows (`overlap_tokens`,
                                        always at least one shared paragraph)
5. Optionally prepend the breadcrumb to the embedded text so a chunk that just
   says "It defaults to `None`." still carries "Query Parameters > Optional
   parameters" context.

Token budgets (full rationale in docs/architecture.md §3.1):
  max_tokens=450     bge-small-en-v1.5 truncates hard at 512; the breadcrumb eats the rest.
  min_tokens=64      below this a chunk is usually a stub heading with no standalone value.
  overlap_tokens=60  ~1-2 sentences; only paid on forced splits, not on every chunk.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from docqa.config import ChunkingConfig
from docqa.ingestion.loaders.base import RawDoc

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```+|~~~+)")
_CODE_BLOCK = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)


# --------------------------------------------------------------------------- #
# Token counting                                                               #
# --------------------------------------------------------------------------- #
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """No-dependency approximation (~1.3 tokens/word). Used in tests and as a
    fallback when transformers can't be imported."""

    def count(self, text: str) -> int:
        return int(len(text.split()) * 1.3) + 1


class HFTokenCounter:
    """Exact bge-small token counts via the model's tokenizer (few MB, no model
    weights loaded)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=True))


def default_token_counter(model_name: str) -> TokenCounter:
    try:
        return HFTokenCounter(model_name)
    except Exception:  # offline / transformers missing -> still usable
        return HeuristicTokenCounter()


# --------------------------------------------------------------------------- #
# Data types                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Section:
    breadcrumb: list[str]   # e.g. ["Tutorial", "Query Parameters"]
    level: int              # heading depth of the last crumb (0 = preamble before any heading)
    text: str               # heading line + body, stripped


@dataclass(slots=True)
class Chunk:
    corpus: str
    source_id: str
    url: str
    title: str
    section_path: list[str]
    ordinal: int
    text: str
    token_count: int
    embed_text: str = field(default="", compare=False)

    @property
    def id(self) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{self.corpus}:{self.source_id}:{self.ordinal}")


# --------------------------------------------------------------------------- #
# Step 1-2: markdown -> sections                                               #
# --------------------------------------------------------------------------- #
def split_into_sections(markdown: str, root_path: list[str]) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []          # (level, heading_text)
    buf: list[str] = []
    cur_level = 0
    in_fence = False
    fence_char = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([*root_path, *(h for _, h in stack)], cur_level, body))

    for line in markdown.splitlines():
        fence = _FENCE.match(line)
        if fence:
            if not in_fence:
                in_fence, fence_char = True, fence.group(1)[0]
            elif line.strip().startswith(fence_char * 3):
                in_fence = False
            buf.append(line)
            continue

        heading = None if in_fence else _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            cur_level = level
            buf = [line]                        # keep the heading line inside the body
        else:
            buf.append(line)

    flush()
    return sections


# --------------------------------------------------------------------------- #
# Step 3: merge stubs                                                          #
# --------------------------------------------------------------------------- #
def _prose_tokens(text: str, counter: TokenCounter) -> int:
    """Token count with fenced code blocks removed. A section with lots of code but
    little prose (a `## Run the App` block: heading + one line + a shell snippet)
    reads as substantial by raw length, but its embedding is dominated by code
    tokens that match natural-language questions poorly and it carries no
    explanation — so it's a bad standalone retrieval unit."""
    return counter.count(_CODE_BLOCK.sub(" ", text))


def _needs_merge(sec: Section, counter: TokenCounter, min_tokens: int) -> bool:
    return (
        counter.count(sec.text) < min_tokens
        or _prose_tokens(sec.text, counter) < min_tokens
    )


def _deeper_breadcrumb(preferred: list[str], other: list[str]) -> list[str]:
    """The more specific (deeper) of the two; `preferred` wins a tie."""
    return preferred if len(preferred) >= len(other) else other


def _join_sections(first: Section, second: Section, breadcrumb: list[str]) -> Section:
    return Section(breadcrumb, min(first.level, second.level), first.text + "\n\n" + second.text)


def _merge_small_sections(
    sections: list[Section], counter: TokenCounter, min_tokens: int, max_tokens: int
) -> list[Section]:
    merged: list[Section] = []
    for sec in sections:
        tok = counter.count(sec.text)
        if (
            merged
            and _needs_merge(sec, counter, min_tokens)
            and counter.count(merged[-1].text) + tok <= max_tokens
        ):
            prev = merged[-1]
            merged[-1] = _join_sections(prev, sec, _deeper_breadcrumb(prev.breadcrumb, sec.breadcrumb))
        else:
            merged.append(sec)

    # a tiny/prose-poor leading section has no predecessor -> merge it forward instead
    if len(merged) >= 2 and _needs_merge(merged[0], counter, min_tokens):
        a, b = merged[0], merged[1]
        if counter.count(a.text) + counter.count(b.text) <= max_tokens:
            merged[1] = _join_sections(a, b, _deeper_breadcrumb(b.breadcrumb, a.breadcrumb))
            merged.pop(0)
    return merged


# --------------------------------------------------------------------------- #
# Step 4: sections -> chunks                                                   #
# --------------------------------------------------------------------------- #
def _atomize(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Break text into pieces each <= max_tokens: paragraphs first, then sentences
    inside an oversized paragraph, then whitespace-delimited words inside an
    oversized sentence (a long code line, a URL-heavy run). Guarantees no atom
    can blow the embedding model's context on its own."""
    atoms: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if counter.count(para) <= max_tokens:
            atoms.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            if counter.count(sent) <= max_tokens:
                atoms.append(sent)
                continue
            cur: list[str] = []
            for word in sent.split():
                cur.append(word)
                if counter.count(" ".join(cur)) >= max_tokens:
                    atoms.append(" ".join(cur))
                    cur = []
            if cur:
                atoms.append(" ".join(cur))
    return atoms


def _split_oversized(
    text: str, counter: TokenCounter, max_tokens: int, overlap_tokens: int
) -> list[str]:
    paras = _atomize(text, counter, max_tokens)
    windows: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    for para in paras:
        ptok = counter.count(para)
        if cur and cur_tok + ptok > max_tokens:
            windows.append("\n\n".join(cur))
            # Carry trailing atoms as overlap. Keep at least one *if it still leaves
            # room for the incoming atom*; past that, stop at the overlap_tokens
            # target. This prevents the overlap from doubling a window when atoms
            # are themselves ~max_tokens (big code blocks).
            room_for_overlap = max_tokens - ptok
            carry: list[str] = []
            carry_tok = 0
            for prev in reversed(cur):
                t = counter.count(prev)
                if not carry:
                    if t > room_for_overlap:
                        break
                elif carry_tok + t > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tok += t
            cur, cur_tok = carry[:], carry_tok
        cur.append(para)
        cur_tok += ptok

    if cur:
        windows.append("\n\n".join(cur))
    return windows


def chunk_document(
    doc: RawDoc, corpus: str, cfg: ChunkingConfig, counter: TokenCounter
) -> list[Chunk]:
    sections = _merge_small_sections(
        split_into_sections(doc.body_markdown, doc.section_path),
        counter,
        cfg.min_tokens,
        cfg.max_tokens,
    )
    chunks: list[Chunk] = []

    def emit(text: str, breadcrumb: list[str]) -> None:
        text = text.strip()
        if not text:
            return
        crumb = " > ".join(breadcrumb)
        embed_text = f"{crumb}\n\n{text}" if cfg.prepend_breadcrumb and crumb else text
        chunks.append(
            Chunk(
                corpus=corpus,
                source_id=doc.source_id,
                url=doc.url,
                title=doc.title,
                section_path=breadcrumb,
                ordinal=len(chunks),
                text=text,
                token_count=counter.count(text),
                embed_text=embed_text,
            )
        )

    for sec in sections:
        if counter.count(sec.text) <= cfg.max_tokens:
            emit(sec.text, sec.breadcrumb)
        else:
            for window in _split_oversized(sec.text, counter, cfg.max_tokens, cfg.overlap_tokens):
                emit(window, sec.breadcrumb)

    return chunks
