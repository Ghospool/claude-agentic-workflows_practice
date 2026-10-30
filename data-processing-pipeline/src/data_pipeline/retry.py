from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from agentic_core import is_transient

T = TypeVar("T")


async def retry_async(fn: Callable[[], Awaitable[T]], *, attempts: int = 3, base_delay: float = 0.5,
                      max_delay: float = 20.0, on_retry: Callable[[int, BaseException], None] | None = None) -> T:
    """Exponential backoff with full jitter; only transient errors are retried."""
    for n in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if n == attempts or not is_transient(exc):
                raise
            if on_retry:
                on_retry(n, exc)
            await asyncio.sleep(random.uniform(0, min(max_delay, base_delay * 2 ** (n - 1))))
    raise AssertionError("unreachable")
