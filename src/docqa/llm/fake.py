"""Deterministic stand-in LLM for tests and offline graph runs.

Branches on `json_mode`: the router calls with json_mode=True, sub-agents don't.
Pass `handler` for full control, or `route_response` / `answer` for the common case.
"""

from __future__ import annotations

from collections.abc import Callable

from docqa.llm.base import ChatMessage


class FakeLLM:
    name = "fake"

    def __init__(
        self,
        *,
        route_response: str = '{"category": "fallback", "confidence": 0.0}',
        answer: str = "Stub answer grounded in [1].",
        handler: Callable[..., str] | None = None,
    ) -> None:
        self.route_response = route_response
        self.answer = answer
        self.handler = handler
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(
            {"messages": messages, "temperature": temperature, "json_mode": json_mode}
        )
        if self.handler is not None:
            return self.handler(messages, temperature=temperature, json_mode=json_mode)
        return self.route_response if json_mode else self.answer
