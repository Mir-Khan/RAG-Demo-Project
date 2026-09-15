"""Google Gemini via the google-genai SDK. Free tier: https://aistudio.google.com/apikey

Gemini's chat format differs from the OpenAI shape the other providers use:
  * there is no `system` turn — system text is a separate `system_instruction`
  * the assistant role is called `model`
We translate our provider-neutral ChatMessage list here so prompts stay portable.
"""

from __future__ import annotations

from docqa.llm._retry import with_retry
from docqa.llm.base import ChatMessage

# `-latest` alias so it doesn't 404 when Google rotates versions. The "lite" line
# has the roomiest free-tier limits (higher RPM/RPD) and is plenty for grounded
# RAG extraction; override with LLM_MODEL in .env for a stronger model.
_DEFAULT_MODEL = "gemini-flash-lite-latest"


class GeminiLLM:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "") -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is empty. Get one at https://aistudio.google.com/apikey "
                "or set LLM_PROVIDER to groq / ollama."
            )
        from google import genai

        self.model = model or _DEFAULT_MODEL
        self._client = genai.Client(api_key=api_key)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        from google.genai import types

        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        cfg: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "system_instruction": system,
        }
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        # NB: 2.5/3.x "flash" models "think" before answering and the reasoning
        # tokens count against max_output_tokens — callers must budget generously
        # (see the router / answer_max_tokens sizes). Explicitly disabling thinking
        # (thinking_budget=0) is rejected (400) by the 3.x flash models, so we don't.
        config = types.GenerateContentConfig(**cfg)

        resp = with_retry(
            lambda: self._client.models.generate_content(
                model=self.model, contents=turns, config=config
            )
        )
        text = (resp.text or "").strip()
        if not text:
            reason = None
            if getattr(resp, "candidates", None):
                reason = getattr(resp.candidates[0], "finish_reason", None)
            raise RuntimeError(
                f"Gemini ({self.model}) returned no text (finish_reason={reason}). "
                "If this is MAX_TOKENS, raise ANSWER_MAX_TOKENS; if SAFETY, rephrase."
            )
        return text
