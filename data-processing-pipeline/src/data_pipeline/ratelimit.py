"""Async token-bucket limiter + concurrency gate wrapped around any LLM client."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from agentic_core import LLMClient, LLMResponse, Usage


class TokenBucket:
    def __init__(self, rate_per_s: float, capacity: int | None = None) -> None:
        self.rate = rate_per_s
        self.capacity = capacity or max(1, int(rate_per_s))
        self._tokens = float(self.capacity)
        self._t = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._t) * self.rate)
                self._t = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            await asyncio.sleep(wait)


class Gateway:
    """Every API call in the pipeline goes through here: bounded in-flight calls + request rate."""

    def __init__(self, client: LLMClient, concurrency: int, rate_per_s: float) -> None:
        self._client = client
        self._sem = asyncio.Semaphore(concurrency)
        self._bucket = TokenBucket(rate_per_s)
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0

    @property
    def total_usage(self) -> Usage:
        return self._client.total_usage

    async def complete(self, **kw: Any) -> LLMResponse:
        async with self._sem:
            await self._bucket.acquire()
            self.calls += 1
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            try:
                return await self._client.complete(**kw)
            finally:
                self._in_flight -= 1
