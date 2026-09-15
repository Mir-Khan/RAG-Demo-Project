"""LLM provider factory."""

from __future__ import annotations

from docqa.config import Settings, get_settings
from docqa.llm.base import ChatMessage, LLM

__all__ = ["ChatMessage", "LLM", "get_llm"]


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from docqa.llm.gemini_provider import GeminiLLM

        return GeminiLLM(settings.google_api_key, settings.llm_model)
    if provider == "groq":
        from docqa.llm.groq_provider import GroqLLM

        return GroqLLM(settings.groq_api_key, settings.llm_model)
    if provider == "ollama":
        from docqa.llm.ollama_provider import OllamaLLM

        return OllamaLLM(settings.ollama_base_url, settings.llm_model)
    if provider == "fake":
        from docqa.llm.fake import FakeLLM

        return FakeLLM()
    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r} (gemini | groq | ollama | fake)"
    )
