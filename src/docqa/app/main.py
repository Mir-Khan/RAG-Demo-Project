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
from docqa.config import CORPORA_DIR, REPO_ROOT, get_settings, load_corpus_config

_AVATAR_DIR = REPO_ROOT / "assets" / "avatars" / "pokemon"
_AVATAR_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _fan_art(name: str) -> str | None:
    """Path to assets/avatars/pokemon/<name>.<ext> if someone's dropped one in,
    else None — the caller falls back to an emoji. Never raises: an empty or
    missing folder is the expected default state, not an error."""
    for ext in _AVATAR_EXTS:
        path = _AVATAR_DIR / f"{name}{ext}"
        if path.is_file():
            return str(path)
    return None

# Fly (and most container platforms) capture stdout as the app's logs — this is
# the only record of what actually happened once a session's browser tab is
# closed. A silently-wrong answer (no exception, just a bad refusal) is
# otherwise undiagnosable after the fact, as it was the first time this
# happened: nothing was logged, so root-causing it meant re-deriving the whole
# request from a screenshot and a live reproduction.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("docqa.app")

st.set_page_config(page_title="Docs Q&A", page_icon="📘", layout="centered")

# Base polish applied regardless of corpus: tighter expander/divider spacing,
# slightly rounder buttons, a bit more breathing room around the chat log.
_BASE_CSS = """
<style>
  .block-container { padding-top: 2.5rem; max-width: 780px; }
  .stButton > button { border-radius: 8px; }
  div[data-testid="stExpander"] { border-radius: 10px; }
  div[data-testid="stChatMessage"] { padding-bottom: 0.15rem; }
</style>
"""

# Streamlit's theme file (.streamlit/config.toml) sets the FastAPI-tuned default
# — clean, minimal, calm blue — and can't change per session. This overrides just
# the accent color + button/link styling at runtime for a livelier feel when the
# Pokémon corpus is picked, without touching the base theme file.
_POKEMON_CSS = """
<style>
  :root { --primary-color: #EE1515; }
  .stButton > button[kind="secondary"] { border: 1px solid #EE1515; }
  a { color: #FFCB05 !important; }
  .stButton > button:hover { border-color: #EE1515; color: #EE1515; }
</style>
"""

_CORPUS_STYLE = {
    "fastapi": {"sidebar_icon": "📘", "assistant_avatar": "📘", "user_avatar": "🧑‍💻"},
    "pokemon": {"sidebar_icon": "⚡", "assistant_avatar": "⚡", "user_avatar": "🎮"},
}
_DEFAULT_STYLE = {"sidebar_icon": "📘", "assistant_avatar": "🤖", "user_avatar": "🧑"}


def _style_for(corpus: str) -> dict:
    style = dict(_CORPUS_STYLE.get(corpus, _DEFAULT_STYLE))
    if corpus == "pokemon":
        # fan-art avatars override the emoji defaults the moment they exist —
        # see assets/avatars/pokemon/README.md
        style["assistant_avatar"] = _fan_art("assistant") or style["assistant_avatar"]
        style["user_avatar"] = _fan_art("user") or style["user_avatar"]
    return style


@st.cache_resource(show_spinner="Loading embedder…")
def _embedder():
    from docqa.ingestion.embed import Embedder

    settings = get_settings()
    return Embedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        expected_dim=settings.embedding_dim,
    )


@st.cache_resource(show_spinner="Loading reranker…")
def _reranker():
    from docqa.retrieval.rerank import Reranker

    return Reranker(get_settings().reranker_model)


@st.cache_resource(show_spinner="Connecting…")
def _pipeline(corpus: str):
    from docqa.agents.graph import QAPipeline
    from docqa.retrieval.retriever import Retriever

    # bge-small and the reranker are the same weights regardless of corpus --
    # share one copy across corpus switches instead of loading a second set
    # into memory the moment someone tries both in one session.
    retriever = Retriever(corpus, embedder=_embedder(), reranker=_reranker())
    return QAPipeline(corpus=corpus, retriever=retriever)


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
st.markdown(_BASE_CSS, unsafe_allow_html=True)

with st.sidebar:
    corpus = st.selectbox("Corpus", _corpus_names())
    style = _style_for(corpus)
    st.title(f"{style['sidebar_icon']} Docs Q&A")
    if corpus == "pokemon":
        st.caption(
            "Fan-made demo project — not affiliated with, endorsed by, or sponsored by "
            "Nintendo, Game Freak, Creatures Inc., or The Pokémon Company. Pokémon is a "
            "trademark of Nintendo."
        )
    cfg = load_corpus_config(corpus)
    settings = get_settings()
    st.caption(f"LLM · `{settings.llm_provider}` · `{settings.llm_model}`")
    st.caption("Retrieval config")
    st.json(cfg.retrieval.model_dump(), expanded=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if corpus == "pokemon":
    st.markdown(_POKEMON_CSS, unsafe_allow_html=True)

st.session_state.setdefault("history", [])

# --------------------------------------------------------------------------- #
# transcript                                                                   #
# --------------------------------------------------------------------------- #
for turn in st.session_state.history:
    avatar = style["user_avatar"] if turn["role"] == "user" else style["assistant_avatar"]
    with st.chat_message(turn["role"], avatar=avatar):
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
