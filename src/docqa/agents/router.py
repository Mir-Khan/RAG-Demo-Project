"""Query classification + its failure handling.

The router only ever *degrades quality*, never correctness: on a bad label, low
confidence, or unparseable output we fall back to a generic sub-agent that runs
the same retrieval. So parsing is deliberately forgiving.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

FALLBACK = "fallback"
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class RouteDecision:
    category: str          # a known id, or "fallback"
    confidence: float
    fallback_used: bool
    raw: str


def parse_route(raw: str, known_ids: set[str], min_confidence: float) -> RouteDecision:
    match = _JSON_OBJ.search(raw or "")
    if not match:
        return RouteDecision(FALLBACK, 0.0, True, raw)
    try:
        data = json.loads(match.group(0))
        category = str(data.get("category", "")).strip()
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        return RouteDecision(FALLBACK, 0.0, True, raw)

    if category not in known_ids:
        return RouteDecision(FALLBACK, confidence, True, raw)
    if confidence < min_confidence:
        return RouteDecision(FALLBACK, confidence, True, raw)
    return RouteDecision(category, confidence, False, raw)
