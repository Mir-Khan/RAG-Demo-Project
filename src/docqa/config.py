"""Configuration: process settings from env (.env) + per-corpus YAML.

Two layers, deliberately separate:
  * Settings    — infra/secrets that change per *environment* (DB URL, API keys, model ids).
  * CorpusConfig — everything that changes per *knowledge base* (loader, chunking,
                   router categories). The rest of the codebase never hardcodes a
                   corpus name; it asks for `load_corpus_config(settings.corpus)`.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPORA_DIR = REPO_ROOT / "config" / "corpora"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://docqa:docqa@localhost:5432/docqa"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # bge was trained with an instruction on the QUERY side only; passages get no prefix.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages:"

    reranker_model: str = "BAAI/bge-reranker-base"

    # LLM: provider-abstracted. "gemini" (default) | "groq" | "ollama" (local) | "fake" (tests)
    llm_provider: str = "gemini"
    google_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = ""               # empty -> each provider's own sensible default
    llm_temperature: float = 0.2      # fallback agent + default
    answer_max_tokens: int = 1500     # headroom: "thinking" models spend some of this before visible text

    # Orchestration
    router_min_confidence: float = 0.45   # below this -> fallback agent
    max_context_tokens: int = 2800        # budget for retrieved chunks in the prompt (~final_k * max_tokens)

    # Evaluation. The RAGAS judge follows llm_provider; empty model -> provider default.
    # Judge should ideally differ from the generator to reduce self-preference bias;
    # kept the same here for $0. Pin the model in .env so scores compare across runs.
    eval_judge_model: str = ""

    corpus: str = "fastapi"


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# Per-corpus config                                                            #
# --------------------------------------------------------------------------- #
class LoaderConfig(BaseModel):
    type: str  # key into docqa.ingestion.loaders.build_loader

    # github_markdown
    repo: str | None = None
    ref: str | None = None
    include_globs: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)
    url_base: str = ""
    strip_path_prefix: str = ""

    # markdown_dir
    path: str | None = None


class ChunkingConfig(BaseModel):
    max_tokens: int = 450       # bge-small truncates at 512; leave room for the breadcrumb
    min_tokens: int = 64        # sections shorter than this get merged into a neighbour
    overlap_tokens: int = 60    # only applied when one section must be split across windows
    prepend_breadcrumb: bool = True


class RetrievalConfig(BaseModel):
    """Tunable knobs for the hybrid retrieval pipeline. The eval harness (stage 4)
    sweeps these to produce the baseline-vs-tuned results table."""

    dense_k: int = 30           # candidates pulled from vector search
    sparse_k: int = 30          # candidates pulled from full-text search
    rrf_k: int = 60             # Reciprocal Rank Fusion constant; 60 is the standard default
    rerank: bool = True         # run the cross-encoder reranker?
    rerank_top_n: int = 20      # how many fused candidates the reranker scores
    final_k: int = 6            # chunks actually handed to the LLM


class RouterCategory(BaseModel):
    id: str
    description: str
    examples: list[str] = Field(default_factory=list)
    # how the sub-agent for this category should answer (appended to the system prompt)
    answer_style: str = ""
    # generation temperature for this category's sub-agent
    temperature: float = 0.1


class CorpusConfig(BaseModel):
    name: str
    display_name: str = ""
    loader: LoaderConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    router_categories: list[RouterCategory] = Field(default_factory=list)
    # Demo-only: question(s) the corpus genuinely doesn't answer, surfaced in the UI
    # as a one-click example so the honest-refusal behavior is easy to show off
    # rather than something a visitor has to stumble into by accident. Optional —
    # an empty list just hides that part of the UI (e.g. the Pokémon demo corpus).
    demo_negative_examples: list[str] = Field(default_factory=list)


@functools.lru_cache
def load_corpus_config(name: str | None = None) -> CorpusConfig:
    name = name or get_settings().corpus
    path = CORPORA_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus config at {path}. Available: "
            f"{sorted(p.stem for p in CORPORA_DIR.glob('*.yaml'))}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CorpusConfig.model_validate(data)
