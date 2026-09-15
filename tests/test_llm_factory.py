"""get_llm() provider selection + key validation. No SDKs or network needed —
every provider checks its key before importing its client library."""

from __future__ import annotations

import pytest

from docqa.config import Settings
from docqa.llm import get_llm


def _settings(**kw) -> Settings:
    # _env_file=None makes this hermetic even if the repo has a real .env
    return Settings(_env_file=None, **kw)


def test_fake_provider_needs_nothing():
    assert get_llm(_settings(llm_provider="fake")).name == "fake"


def test_gemini_without_key_raises_clearly():
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_llm(_settings(llm_provider="gemini", google_api_key=""))


def test_groq_without_key_raises_clearly():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm(_settings(llm_provider="groq", groq_api_key=""))


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm(_settings(llm_provider="banana"))


def test_default_provider_is_gemini():
    assert _settings().llm_provider == "gemini"
