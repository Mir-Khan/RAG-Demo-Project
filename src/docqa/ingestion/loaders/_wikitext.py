"""MediaWiki wikitext -> markdown-ish plain text.

Deliberately conservative, same philosophy as `_markdown.py`'s MkDocs cleanup:
good enough for the chunker, not a full wikitext parser. Wikitext's template
system ({{...}}) is Turing-complete in the general case; we handle the common,
recognizable shapes and fall back to stripping anything we don't understand
rather than leaving raw markup in the text an embedding model would see.
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TABLE = re.compile(r"^\{\|.*?^\|\}", re.DOTALL | re.MULTILINE)
_FILE_LINK = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)
_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
_WIKILINK = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
_BOLD_ITALIC = re.compile(r"'''''(.*?)'''''")
_BOLD = re.compile(r"'''(.*?)'''")
_ITALIC = re.compile(r"''(.*?)''")
# {{name|a|b|c}} -> last positional arg if any (usually display text), else drop.
# Handles one level of nesting (templates inside template args) via a manual scan.
_TEMPLATE_OPEN = "{{"
_TEMPLATE_CLOSE = "}}"


def _strip_templates(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith(_TEMPLATE_OPEN, i):
            depth = 1
            j = i + 2
            while j < n and depth:
                if text.startswith(_TEMPLATE_OPEN, j):
                    depth += 1
                    j += 2
                elif text.startswith(_TEMPLATE_CLOSE, j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            inner = text[i + 2 : j - 2]
            parts = inner.split("|")
            # {{TemplateName}} alone (no args) is almost always a pure marker
            # (navboxes, {{CURRENTGEN}}, category-style templates) -> drop.
            # {{name|a|b}} usually ends in the human-readable text -> keep it.
            if len(parts) > 1:
                candidate = parts[-1].strip()
                if candidate and not candidate.startswith(("*", "image=", "file=")):
                    out.append(candidate)
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def wikitext_to_markdown(text: str) -> str:
    text = _COMMENT.sub("", text)
    text = _REF.sub("", text)
    text = _TABLE.sub("", text)
    text = _FILE_LINK.sub("", text)
    # wikitext's own "#" is an ordered-list marker, not a heading -- convert it
    # to a plain bullet *before* turning "==Heading==" into "# Heading" below,
    # or a stray list item reads as a section break to our chunker.
    text = re.sub(r"^#+\s*", "- ", text, flags=re.MULTILINE)
    text = _HEADING.sub(lambda m: "#" * len(m.group(1)) + " " + m.group(2), text)
    text = _strip_templates(text)
    text = _WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = _BOLD_ITALIC.sub(r"**\1**", text)
    text = _BOLD.sub(r"**\1**", text)
    text = _ITALIC.sub(r"*\1*", text)
    # bullet/definition markers -> plain markdown bullets
    text = re.sub(r"^\*+\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^:+\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
