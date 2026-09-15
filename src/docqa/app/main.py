"""Streamlit UI for the multi-agent documentation Q&A pipeline.

    streamlit run src/docqa/app/main.py        (or: make app)

Thin view over QAPipeline.answer(): a question goes in, a cited answer comes out,
and a transparency panel shows every retrieved chunk with the score it earned at
each retrieval stage and whether the answer actually cited it.
"""

from __future__ import annotations

import logging

import streamlit as st

from docqa.app.components import score_chips, source_status, truncate
from docqa.config import CORPORA_DIR, get_settings, load_corpus_config

# Fly (and most container platforms) capture stdout as the app's logs — this is
# the only record of what actually happened once a session's browser tab is
# closed. A silently-wrong answer (no exception, just a bad refusal) is
# otherwise undiagnosable after the fact, as it was the first time this
# happened: nothing was logged, so root-causing it meant re-deriving the whole
# request from a screenshot and a live reproduction.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("docqa.app")

st.set_page_config(page_title="Docs Q&A", page_icon="📘", layout="centered")


@st.cache_resource(show_spinner="Loading embedder + reranker…")
def _pipeline(corpus: str):
    from docqa.agents.graph import QAPipeline

    return QAPipeline(corpus=corpus)


def _corpus_names() -> list[str]:
    return sorted(p.stem for p in CORPORA_DIR.glob("*.yaml"))


def _example_questions(cfg) -> list[str]:
    return [c.examples[0] for c in cfg.router_categories if c.examples]


def _render_meta(result) -> None:
    route = (
        "⚠️ **fallback agent**"
        if result.fallback_used
        else f"routed → **{result.category}** ({result.confidence:.2f})"
    )
    timings = " · ".join(f"{k} {v:.0f} ms" for k, v in result.timings_ms.items())
    st.caption(f"{route}  ·  {timings}")

    trace = result.trace
    if trace is None or not trace.results:
        st.warning(
            "No chunks retrieved — is this corpus ingested? "
            "Run `python -m docqa.ingestion.pipeline --fresh`."
        )
        return

    n_ctx = len(result.citations)
    with st.expander(f"🔎 Sources · {len(trace.results)} retrieved, {n_ctx} in context"):
        for i, chunk in enumerate(trace.results):
            label, status = source_status(i, n_ctx, result.used_citation_numbers)
            head = f"`{label}` " if label else ""
            st.markdown(f"{head}[{chunk.breadcrumb or chunk.title}]({chunk.url}) — *{status}*")
            chips = score_chips(chunk)
            if chips:
                st.caption("  ·  ".join(chips))
            st.markdown(f"> {truncate(chunk.text)}")
            if i < len(trace.results) - 1:
                st.divider()


def _handle(question: str, corpus: str) -> None:
    st.session_state.history.append({"role": "user", "content": question})
    try:
        pipe = _pipeline(corpus)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        logger.exception("pipeline construction failed | corpus=%s", corpus)
        st.session_state.history.append(
            {"role": "assistant", "content": f":red[Could not start the pipeline — {exc}]"}
        )
        return
    try:
        with st.spinner("route → retrieve → answer…"):
            result = pipe.answer(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline error | corpus=%s | question=%r", corpus, question)
        st.session_state.history.append(
            {"role": "assistant", "content": f":red[Pipeline error — {exc}]"}
        )
        return

    trace = result.trace
    top = trace.results[0] if trace and trace.results else None
    logger.info(
        "answered | corpus=%s | question=%r | category=%s%s | conf=%.2f | "
        "n_chunks=%d | top=%r@%.3f | timings=%s | answer_len=%d",
        corpus,
        question,
        result.category,
        "(fallback)" if result.fallback_used else "",
        result.confidence,
        len(trace.results) if trace else 0,
        top.breadcrumb if top else None,
        top.rerank_score if top and top.rerank_score is not None else -1.0,
        result.timings_ms,
        len(result.answer or ""),
    )
    st.session_state.history.append(
        {"role": "assistant", "content": result.answer or "_(no answer)_", "result": result}
    )


# --------------------------------------------------------------------------- #
# sidebar                                                                      #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📘 Docs Q&A")
    corpus = st.selectbox("Corpus", _corpus_names())
    cfg = load_corpus_config(corpus)
    settings = get_settings()
    st.caption(f"LLM · `{settings.llm_provider}` · `{settings.llm_model}`")
    st.caption("Retrieval config")
    st.json(cfg.retrieval.model_dump(), expanded=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.session_state.setdefault("history", [])

# --------------------------------------------------------------------------- #
# transcript                                                                   #
# --------------------------------------------------------------------------- #
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("result") is not None:
            _render_meta(turn["result"])

if not st.session_state.history:
    st.markdown(f"#### Ask about the **{cfg.display_name or cfg.name}**")
    cols = st.columns(2)
    for i, ex in enumerate(_example_questions(cfg)):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending_q = ex
            st.rerun()

    if cfg.demo_negative_examples:
        st.caption("Or watch it correctly decline — a question genuinely outside the docs:")
        neg = cfg.demo_negative_examples[0]
        if st.button(f"🚫 {neg}", key="ex_negative", use_container_width=True):
            st.session_state.pending_q = neg
            st.rerun()

# --------------------------------------------------------------------------- #
# input                                                                        #
# --------------------------------------------------------------------------- #
pending = st.session_state.pop("pending_q", None)
typed = st.chat_input("Ask about the docs…")
question = typed or pending
if question:
    _handle(question, corpus)
    st.rerun()
