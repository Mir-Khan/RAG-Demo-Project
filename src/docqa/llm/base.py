"""The LLM seam.

One method, `complete(messages, ...) -> str`. Prompts stay provider-neutral so
swapping Groq <-> Ollama <-> a paid endpoint is a config change, never a rewrite.
`json_mode` asks the provider to constrain output to a JSON object (used by the
router); providers that can't enforce it should still try via prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str


@runtime_checkable
class LLM(Protocol):
    name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str: ...


def to_openai_messages(messages: list[ChatMessage]) -> list[dict]:
    """Groq and Ollama both speak the OpenAI chat-message shape."""
    return [{"role": m.role, "content": m.content} for m in messages]
