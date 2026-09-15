"""Local Ollama chat. Zero API dependency, no rate limits — the offline fallback.

Slower on CPU and too heavy for a low-tier deploy box, so it's the local-dev /
demo option rather than the deployed default.
"""

from __future__ import annotations

import httpx

from docqa.llm.base import ChatMessage, to_openai_messages


class OllamaLLM:
    name = "ollama"

    def __init__(self, base_url: str, model: str = "", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or "llama3.1"
        self._client = httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
