"""Small markdown helpers shared by loaders.

Deliberately conservative: we normalise a few MkDocs-flavoured constructs that
would otherwise become noise in retrieval, and leave everything else untouched.
"""

from __future__ import annotations

import re

_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
_INCLUDE = re.compile(r"^\s*\{[!*].*[!*]\}\s*$", re.MULTILINE)         # {!../snippet!}  and  {* ../snippet *}
_TAB_FENCE = re.compile(r"^\s*/{4}\s*(tab\s*\|.*)?$", re.MULTILINE)     # //// tab | Title   and closing ////
_ADMONITION = re.compile(r'^(\s*)[!?]{3}\s+\w+(?:\s+"([^"]*)")?\s*$', re.MULTILINE)


def first_heading(text: str) -> str | None:
    m = _H1.search(text)
    return m.group(1).strip() if m else None


def light_clean(text: str) -> str:
    """Strip include directives and tab fences; flatten admonition markers to bold."""
    text = _INCLUDE.sub("", text)
    text = _TAB_FENCE.sub("", text)
    text = _ADMONITION.sub(lambda m: f'{m.group(1)}**{(m.group(2) or "Note").strip()}**', text)
    # collapse 3+ blank lines left behind
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
