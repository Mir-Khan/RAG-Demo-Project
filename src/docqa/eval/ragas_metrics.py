"""RAGAS integration — the ONLY file that imports ragas.

RAGAS's public API moves between minor versions, so it is quarantined here: if a
future version renames something, this is the one place to fix, and the rest of
the harness (deterministic metrics, runner, diff) is unaffected.

Written against ragas 0.2.x. Metrics used, all reference-free (no gold answer):
  * Faithfulness                       - claims supported by retrieved context
  * ResponseRelevancy                  - answer actually addresses the question
  * LLMContextPrecisionWithoutReference - retrieved chunks relevant & well-ranked

Judge LLM: whichever provider the run selects (Ollama recommended — RAGAS makes
3+ judge calls per item, too many for a hosted free tier across a tuning loop).
Embeddings for ResponseRelevancy: local bge-small, so that side is always $0.
"""

from __future__ import annotations

import sys
import types


def _stub_missing_vertexai() -> None:
    """ragas (<=0.4.3) hard-imports `langchain_community.chat_models.vertexai.ChatVertexAI`
    at module load. langchain-community >=0.4 removed that path (Vertex moved to
    `langchain-google-vertexai`), and we can't downgrade langchain-community without
    breaking langgraph. We never use Vertex, so register a stub that lets `import
    ragas` succeed. Instantiating the stub raises — which is correct, nothing should."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
        return  # real module exists on this install
    except ModuleNotFoundError:
        pass

    mod = types.ModuleType(name)

    class ChatVertexAI:  # noqa: D401 - stand-in
        def __init__(self, *_, **__):
            raise RuntimeError(
                "ChatVertexAI is stubbed out. Install langchain-google-vertexai to use Vertex."
            )

    mod.ChatVertexAI = ChatVertexAI
    sys.modules[name] = mod
    try:
        import langchain_community.chat_models as _cm

        _cm.vertexai = mod
    except Exception:  # noqa: BLE001
        pass


# friendly, stable names we report regardless of ragas internals
_FRIENDLY = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "response_relevancy": "answer_relevancy",
    "llm_context_precision_without_reference": "context_precision",
    "context_precision": "context_precision",
}


def available() -> bool:
    try:
        _stub_missing_vertexai()
        import langchain_huggingface  # noqa: F401
        import ragas  # noqa: F401

        return True
    except Exception:
        return False


def _judge_chat_model(
    provider: str, judge_model: str, groq_api_key: str, google_api_key: str, ollama_base_url: str
):
    """A langchain chat model for RAGAS to use as its judge, matching the provider.

    Ollama is the sane choice here: RAGAS fires 3+ judge calls per eval item, which
    a hosted free tier (5 RPM / 20-1000 RPD) cannot sustain across a tuning loop.
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=judge_model or "llama3.1", base_url=ollama_base_url, temperature=0.0
        )
    if provider == "gemini":
        if not google_api_key:
            raise ValueError("RAGAS judge needs GOOGLE_API_KEY (or run with --no-ragas)")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=judge_model or "gemini-flash-lite-latest",
            google_api_key=google_api_key,
            temperature=0.0,
        )
    if provider == "groq":
        if not groq_api_key:
            raise ValueError("RAGAS judge needs GROQ_API_KEY (or run with --no-ragas)")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=judge_model or "llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.0
        )
    raise ValueError(
        f"RAGAS judge not wired for provider {provider!r} — use ollama, gemini, groq, or --no-ragas"
    )


def score_with_ragas(
    rows: list[dict],
    *,
    provider: str,
    judge_model: str,
    groq_api_key: str = "",
    google_api_key: str = "",
    ollama_base_url: str = "http://localhost:11434",
    embedding_model: str,
) -> tuple[dict, list[dict]]:
    """Returns (aggregate {friendly_name: mean}, per_row [{friendly_name: score}, ...])
    aligned to `rows` by index. Each row needs: question, answer, contexts."""
    _stub_missing_vertexai()

    import nest_asyncio
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    nest_asyncio.apply()

    chat = _judge_chat_model(provider, judge_model, groq_api_key, google_api_key, ollama_base_url)

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"] or "",
            retrieved_contexts=r["contexts"] or [""],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    judge = LangchainLLMWrapper(chat)
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))

    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=judge),
    ]
    names = [m.name for m in metrics]

    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()

    per_row: list[dict] = [
        {_FRIENDLY.get(n, n): _to_float(r.get(n)) for n in names} for _, r in df.iterrows()
    ]

    agg: dict[str, float] = {}
    for n in names:
        vals = [v for v in (_to_float(x) for x in df[n].tolist()) if v is not None]
        if vals:
            agg[_FRIENDLY.get(n, n)] = round(sum(vals) / len(vals), 4)
    return agg, per_row


def _to_float(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN
