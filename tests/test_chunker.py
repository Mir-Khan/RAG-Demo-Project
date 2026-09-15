"""Chunker behaviour, using the no-dependency heuristic token counter so these
run without network or the ML stack."""

from __future__ import annotations

from docqa.config import ChunkingConfig
from docqa.ingestion.chunker import (
    HeuristicTokenCounter,
    chunk_document,
    split_into_sections,
)
from docqa.ingestion.loaders.base import RawDoc

COUNTER = HeuristicTokenCounter()

SAMPLE = """# Query Parameters

When you declare other function parameters that are not part of the path
parameters, they are automatically interpreted as "query" parameters.

## Optional parameters

The same way, you can declare optional query parameters, by setting their
default to `None`.

```python
# this hash is NOT a markdown heading
@app.get("/items/")
async def read_items(q: str | None = None):
    return {"q": q}
```

## Recap

Short.
"""


def test_sections_follow_heading_hierarchy():
    secs = split_into_sections(SAMPLE, ["Tutorial"])
    crumbs = [s.breadcrumb for s in secs]
    assert ["Tutorial", "Query Parameters"] in crumbs
    assert ["Tutorial", "Query Parameters", "Optional parameters"] in crumbs
    assert ["Tutorial", "Query Parameters", "Recap"] in crumbs


def test_hash_inside_code_fence_is_not_a_heading():
    secs = split_into_sections(SAMPLE, [])
    # the "# this hash is NOT a markdown heading" line must stay inside the
    # Optional parameters section, not spawn its own
    assert not any("this hash" in " ".join(s.breadcrumb).lower() for s in secs)


def test_small_section_merges_into_neighbour():
    cfg = ChunkingConfig(max_tokens=200, min_tokens=40, overlap_tokens=20)
    chunks = chunk_document(
        RawDoc(source_id="tutorial/query-params", title="Query Parameters",
               url="https://example/", body_markdown=SAMPLE, section_path=["Tutorial"]),
        corpus="fastapi", cfg=cfg, counter=COUNTER,
    )
    # "Recap\n\nShort." is well under min_tokens -> must not be its own chunk
    assert all(c.text.strip() != "## Recap\n\nShort." for c in chunks)
    assert any("Short." in c.text for c in chunks)  # but its content survives


def test_code_heavy_stub_merges_into_neighbour():
    # a real prose section, then a "Run the App" stub that's mostly a code block
    md = (
        "# Databases\n\n"
        "You can use SQLModel to talk to a SQL database. It gives you Python "
        "classes that map to tables and full editor support, and it is built on "
        "top of SQLAlchemy and Pydantic so validation comes for free.\n\n"
        "## Run the App\n\n"
        "Run it:\n\n"
        "```console\n$ fastapi dev main.py\nINFO   Started server\nINFO   Waiting\n"
        "INFO   Application startup complete\n```\n"
    )
    cfg = ChunkingConfig(max_tokens=300, min_tokens=80, overlap_tokens=40)
    doc = RawDoc(source_id="sql", title="Databases", url="u", body_markdown=md, section_path=["Tutorial"])
    chunks = chunk_document(doc, "fastapi", cfg, COUNTER)
    # the stub must not be its own chunk...
    assert not any(c.section_path[-1:] == ["Run the App"] and "SQLModel" not in c.text for c in chunks)
    # ...its content still survives, folded in with the prose
    assert any("fastapi dev main.py" in c.text and "SQLModel" in c.text for c in chunks)


def test_giant_single_paragraph_is_hard_split():
    # one paragraph, no blank lines, no sentence punctuation -> must still be broken up
    body = "# Big\n\n" + "word " * 900
    cfg = ChunkingConfig(max_tokens=120, min_tokens=10, overlap_tokens=20)
    chunks = chunk_document(
        RawDoc(source_id="s", title="t", url="u", body_markdown=body),
        corpus="c", cfg=cfg, counter=COUNTER,
    )
    assert len(chunks) >= 8
    # no chunk may exceed the budget by more than one packing step
    assert max(c.token_count for c in chunks) <= cfg.max_tokens * 1.3


def test_oversized_section_splits_with_overlap():
    body = "# Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    cfg = ChunkingConfig(max_tokens=120, min_tokens=10, overlap_tokens=30)
    chunks = chunk_document(
        RawDoc(source_id="big", title="Big", url="u", body_markdown=body),
        corpus="c", cfg=cfg, counter=COUNTER,
    )
    assert len(chunks) > 1
    assert all(c.token_count <= cfg.max_tokens * 1.5 for c in chunks)  # generous ceiling
    # consecutive windows should share at least one paragraph (overlap)
    first_paras = set(chunks[0].text.split("\n\n"))
    second_paras = set(chunks[1].text.split("\n\n"))
    assert first_paras & second_paras


def test_chunk_ids_are_stable_and_unique():
    doc = RawDoc(source_id="tutorial/query-params", title="Q", url="u", body_markdown=SAMPLE)
    a = chunk_document(doc, "fastapi", ChunkingConfig(), COUNTER)
    b = chunk_document(doc, "fastapi", ChunkingConfig(), COUNTER)
    assert [c.id for c in a] == [c.id for c in b]          # deterministic
    assert len({c.id for c in a}) == len(a)                # unique within a doc


def test_breadcrumb_is_prepended_to_embed_text():
    doc = RawDoc(source_id="s", title="t", url="u", body_markdown=SAMPLE, section_path=["Tutorial"])
    chunks = chunk_document(doc, "fastapi", ChunkingConfig(prepend_breadcrumb=True), COUNTER)
    assert chunks[0].embed_text.startswith("Tutorial > Query Parameters")
    assert chunks[0].text != chunks[0].embed_text
