"""End-to-end graph flow with a FakeLLM and a fake retriever — no DB, no models."""

from __future__ import annotations

import pytest

from docqa.agents.graph import QAPipeline
from docqa.llm.fake import FakeLLM
from docqa.retrieval.types import RetrievalTrace, RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, **_):
        return RetrievalTrace(query=query, corpus="fastapi", results=list(self._chunks))


def _chunks(n=3):
    return [
        RetrievedChunk(
            id=f"c{i}", url=f"https://docs/{i}", title=f"Page {i}",
            section_path=["Tutorial", f"S{i}"], text="alpha beta gamma", token_count=3,
        )
        for i in range(1, n + 1)
    ]


def _pipe(route_json: str, answer: str = "Grounded [1][3]."):
    llm = FakeLLM(route_response=route_json, answer=answer)
    pipe = QAPipeline(corpus="fastapi", retriever=FakeRetriever(_chunks()), llm=llm)
    return pipe, llm


def test_routes_to_named_category_node():
    pipe, llm = _pipe('{"category": "conceptual", "confidence": 0.9}')
    res = pipe.answer("How does dependency injection work?")
    assert res.category == "conceptual"
    assert res.fallback_used is False
    # the conceptual sub-agent ran: its configured temperature + answer_style reached the LLM
    conceptual = next(c for c in pipe.corpus_cfg.router_categories if c.id == "conceptual")
    gen_call = llm.calls[-1]
    assert gen_call["temperature"] == pytest.approx(conceptual.temperature)
    assert "Explain the idea" in gen_call["messages"][0].content


def test_low_confidence_uses_fallback_node():
    pipe, llm = _pipe('{"category": "api_reference", "confidence": 0.1}')
    res = pipe.answer("something vague")
    assert res.category == "fallback"
    assert res.fallback_used is True
    assert llm.calls[-1]["temperature"] == pytest.approx(pipe.settings.llm_temperature)


def test_unknown_label_uses_fallback_node():
    pipe, _ = _pipe('{"category": "not_a_category", "confidence": 0.99}')
    res = pipe.answer("q")
    assert res.category == "fallback" and res.fallback_used


def test_answer_and_used_citations_pass_through():
    pipe, _ = _pipe('{"category": "troubleshooting", "confidence": 0.8}', answer="Cause X. Fix [2].")
    res = pipe.answer("why 422?")
    assert res.answer == "Cause X. Fix [2]."
    assert res.used_citation_numbers == [2]
    assert [c.n for c in res.citations] == [1, 2, 3]
    assert res.trace is not None
    assert set(res.timings_ms) == {"route", "retrieve", "generate"}


def test_router_call_uses_json_mode():
    pipe, llm = _pipe('{"category": "conceptual", "confidence": 0.9}')
    pipe.answer("q")
    assert llm.calls[0]["json_mode"] is True      # first call is the router
    assert llm.calls[1]["json_mode"] is False     # second is generation
