"""Groq chat completions. Free tier: https://console.groq.com

Groq's LPU inference is fast (hundreds of tokens/sec), so generation is not the
latency bottleneck. The free tier is rate-limited (RPM/TPM) with no SLA — the
provider seam is the insurance against that.
"""

from __future__ import annotations

from docqa.llm._retry import with_retry
from docqa.llm.base import ChatMessage, to_openai_messages


class GroqLLM:
    name = "groq"

    def __init__(self, api_key: str, model: str = "") -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty. Set it in .env or use LLM_PROVIDER=gemini / ollama.")
        from groq import Groq

        self.model = model or "llama-3.3-70b-versatile"
        self._client = Groq(api_key=api_key)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = with_retry(lambda: self._client.chat.completions.create(**kwargs))
        return (resp.choices[0].message.content or "").strip()
