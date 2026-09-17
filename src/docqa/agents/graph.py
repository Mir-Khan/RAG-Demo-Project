"""The LangGraph orchestration graph.

    route ──▶ retrieve ──▶ (conditional on category) ──▶ one sub-agent ──▶ END

`route` and `retrieve` are shared. Each category in the corpus config becomes its
own sub-agent node with its own system prompt + temperature; plus a `fallback`
node for when routing is unsure. All sub-agents call the same shared Retriever.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from docqa.agents.context import assemble_context
from docqa.agents.prompts import render_gen_system, render_router_system, render_user_turn
from docqa.agents.router import FALLBACK, parse_route
from docqa.agents.state import AnswerResult, Citation, QAState
from docqa.config import CorpusConfig, RouterCategory, Settings, get_settings, load_corpus_config
from docqa.ingestion.chunker import default_token_counter
from docqa.llm import ChatMessage, get_llm
from docqa.llm.base import LLM

if TYPE_CHECKING:  # importing Retriever pulls in psycopg; not needed when one is injected
    from docqa.retrieval.retriever import Retriever

_CITE_RE = re.compile(r"\[(\d+)\]")

# state key -> the name it is reported under in AnswerResult.timings_ms
_TIMING_KEYS = {"t_route_ms": "route", "t_retrieve_ms": "retrieve", "t_generate_ms": "generate"}


def _fallback_category(settings: Settings) -> RouterCategory:
    return RouterCategory(
        id=FALLBACK,
        description="General question that does not fit a specific category.",
        answer_style="Answer directly and concisely, strictly from the sources.",
        temperature=settings.llm_temperature,
    )


class QAPipeline:
    """Holds the compiled graph plus its shared Retriever and LLM. Build once,
    reuse for every question (the models load lazily on first query)."""

    def __init__(
        self,
        corpus: str | None = None,
        *,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus_cfg: CorpusConfig = load_corpus_config(corpus or self.settings.corpus)
        if retriever is None:
            from docqa.retrieval.retriever import Retriever

            retriever = Retriever(self.corpus_cfg.name, settings=self.settings)
        self.retriever = retriever
        self.llm = llm or get_llm(self.settings)
        self._counter = default_token_counter(self.settings.embedding_model)
        self._categories = {c.id: c for c in self.corpus_cfg.router_categories}
        self._categories[FALLBACK] = _fallback_category(self.settings)
        self.graph = self._build()

    # -- nodes -------------------------------------------------------- #
    def _route_node(self, state: QAState) -> dict:
        t = time.perf_counter()
        system = render_router_system(self.corpus_cfg)
        raw = self.llm.complete(
            [ChatMessage("system", system), ChatMessage("user", state["query"])],
            temperature=0.0,
            max_tokens=512,  # JSON is tiny, but "thinking" models need headroom before the visible tokens
            json_mode=True,
        )
        known = {c.id for c in self.corpus_cfg.router_categories}
        decision = parse_route(raw, known, self.settings.router_min_confidence)
        return {
            "category": decision.category,
            "confidence": decision.confidence,
            "fallback_used": decision.fallback_used,
            "t_route_ms": round((time.perf_counter() - t) * 1000, 1),
        }

    def _retrieve_node(self, state: QAState) -> dict:
        t = time.perf_counter()
        trace = self.retriever.search(state["query"])
        context, citations = assemble_context(
            trace.results, self.settings.max_context_tokens, self._counter
        )
        return {
            "trace": trace,
            "context": context,
            "citations": citations,
            "t_retrieve_ms": round((time.perf_counter() - t) * 1000, 1),
        }

    def _make_agent_node(self, category: RouterCategory):
        def node(state: QAState) -> dict:
            t = time.perf_counter()
            messages = [
                ChatMessage("system", render_gen_system(self.corpus_cfg, category)),
                ChatMessage("user", render_user_turn(state["query"], state["context"])),
            ]
            answer = self.llm.complete(
                messages,
                temperature=category.temperature,
                max_tokens=self.settings.answer_max_tokens,
            )
            return {"answer": answer, "t_generate_ms": round((time.perf_counter() - t) * 1000, 1)}

        return node

    def _dispatch(self, state: QAState) -> str:
        cat = state.get("category", FALLBACK)
        return cat if cat in self._categories else FALLBACK

    # -- assembly --------------------------------------------------- #
    def _build(self):
        builder = StateGraph(QAState)
        builder.add_node("route", self._route_node)
        builder.add_node("retrieve", self._retrieve_node)
        for cid, category in self._categories.items():
            builder.add_node(cid, self._make_agent_node(category))

        builder.set_entry_point("route")
        builder.add_edge("route", "retrieve")
        builder.add_conditional_edges(
            "retrieve", self._dispatch, {cid: cid for cid in self._categories}
        )
        for cid in self._categories:
            builder.add_edge(cid, END)
        return builder.compile()

    # -- public --------------------------------------------------- #
    def answer(self, query: str) -> AnswerResult:
        final: QAState = self.graph.invoke({"query": query})
        citations: list[Citation] = final.get("citations", [])
        answer_text = final.get("answer", "")
        used = sorted({int(n) for n in _CITE_RE.findall(answer_text)})
        timings = {name: final[key] for key, name in _TIMING_KEYS.items() if key in final}
        return AnswerResult(
            query=query,
            corpus=self.corpus_cfg.name,
            category=final.get("category", FALLBACK),
            confidence=final.get("confidence", 0.0),
            fallback_used=final.get("fallback_used", True),
            answer=answer_text,
            citations=citations,
            used_citation_numbers=used,
            trace=final.get("trace"),
            timings_ms=timings,
        )


def build_pipeline(corpus: str | None = None, **kw) -> QAPipeline:
    return QAPipeline(corpus, **kw)
