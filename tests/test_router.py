"""Router parsing + fallback behaviour."""

from __future__ import annotations

from docqa.agents.router import FALLBACK, parse_route

KNOWN = {"api_reference", "conceptual", "troubleshooting"}


def test_valid_high_confidence_route():
    d = parse_route('{"category": "conceptual", "confidence": 0.9}', KNOWN, 0.45)
    assert d.category == "conceptual"
    assert not d.fallback_used


def test_unknown_label_falls_back():
    d = parse_route('{"category": "banana", "confidence": 0.99}', KNOWN, 0.45)
    assert d.category == FALLBACK and d.fallback_used


def test_low_confidence_falls_back_but_keeps_score():
    d = parse_route('{"category": "api_reference", "confidence": 0.2}', KNOWN, 0.45)
    assert d.category == FALLBACK
    assert d.fallback_used
    assert d.confidence == 0.2


def test_json_wrapped_in_prose_or_fences():
    d = parse_route('Sure!\n```json\n{"category":"troubleshooting","confidence":0.7}\n```', KNOWN, 0.45)
    assert d.category == "troubleshooting"


def test_garbage_output_falls_back():
    for raw in ("", "not json at all", "{broken", '{"confidence": 0.9}'):
        d = parse_route(raw, KNOWN, 0.45)
        assert d.category == FALLBACK and d.fallback_used
