"""Retry hosted-LLM calls through transient rate limits.

Free tiers cap requests-per-minute (and per-day). A 429 is expected traffic, not
an exception — the architecture doc (§3.13) calls for handling it with backoff.
This does exponential backoff + jitter, and honours an explicit retry delay when
the provider includes one in the error (Gemini's RetryInfo, "Please retry in 22s").

Note: backoff only rescues the per-*minute* limit. If the per-*day* quota is
exhausted, switch model or provider (see LLM_PROVIDER=ollama).
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRYABLE = ("RESOURCE_EXHAUSTED", "429", "rate limit", "overloaded", "UNAVAILABLE", "503")
_DELAY_RE = re.compile(r"retry(?:Delay)?[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 503):
        return True
    return any(marker in str(exc) for marker in _RETRYABLE)


def _explicit_delay(exc: Exception) -> float | None:
    m = _DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else None


def with_retry(fn: Callable[[], T], *, attempts: int = 6, base: float = 2.0, cap: float = 60.0) -> T:
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable / out of attempts
            if not _retryable(exc) or i == attempts - 1:
                raise
            delay = _explicit_delay(exc) or min(cap, base * 2**i)
            time.sleep(delay + random.uniform(0, delay * 0.25))
    raise AssertionError("unreachable")
