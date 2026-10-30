"""Parallel pipeline: stream -> validate/dedupe/redact -> batch -> worker pool -> checkpoint -> reduce.

Patterns: SECTIONING (batches processed concurrently) and VOTING (low-confidence records are re-run
and decided by majority). Resilience: bounded concurrency, rate limiting, retry with jittered backoff,
split-on-failure to isolate poison records, dead-letter queue, resumable checkpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from agentic_core import LLMClient, Settings, get_settings
from agentic_core.security import AuditLog, redact_pii

from .aggregate import aggregate, to_markdown
from .extractor import extract
from .ratelimit import Gateway
from .retry import retry_async
from .schemas import Extraction, Failure, Record, RunSummary
from .store import Store

log = logging.getLogger(__name__)
MAX_TEXT = 4000


@dataclass
class PipelineConfig:
    batch_size: int = 20
    concurrency: int = 8
    requests_per_second: float = 20.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    vote_below: float = 0.7          # confidence under which a record is re-run and put to a vote
    extra_votes: int = 2
    vote_temperature: float = 0.7
    price_in_per_mtok: float | None = None    # set from current pricing to get a cost estimate
    price_out_per_mtok: float | None = None


class DataPipeline:
    def __init__(self, client: LLMClient, config: PipelineConfig | None = None,
                 settings: Settings | None = None) -> None:
        self.cfg = config or PipelineConfig()
        self.s = settings or get_settings()
        self.gw = Gateway(client, self.cfg.concurrency, self.cfg.requests_per_second)
        self.model = self.s.fast_model
        self.audit = AuditLog(self.s.audit_log_path)
        self._retries = 0

    # ---------- ingest ----------
    def _read(self, path: Path, summary: RunSummary) -> Iterator[Record]:
        seen: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                summary.input_records += 1
                try:
                    d = json.loads(line)
                    rid, text = str(d["id"]), str(d["text"]).strip()
                    if not rid or not text:
                        raise ValueError("empty id or text")
                except (ValueError, KeyError, TypeError) as e:
                    summary.rejected += 1
                    if len(summary.rejection_samples) < 20:
                        summary.rejection_samples.append(f"line {n}: {type(e).__name__}")
                    continue
                if rid in seen:
                    summary.duplicates += 1
                    continue
                seen.add(rid)
                red = redact_pii(text[:MAX_TEXT])
                yield Record(rid, red.text, seq=n, pii=red.counts)

    # ---------- per-batch resilience ----------
    def _count_retry(self, n: int, exc: BaseException) -> None:
        self._retries += 1
        log.warning("transient error (attempt %d): %s", n, exc)

    async def _extract_resilient(self, batch: list[Record]) -> tuple[list[Extraction], list[Failure]]:
        """Retry transient errors; on hard failure or empty output split the batch to isolate the culprit."""
        try:
            got = await retry_async(lambda: extract(self.gw, batch, self.model), attempts=self.cfg.max_attempts,
                                    base_delay=self.cfg.backoff_base_s, on_retry=self._count_retry)
        except Exception as exc:  # noqa: BLE001
            return await self._split(batch, f"{type(exc).__name__}: {exc}"[:200])
        missing = [r for r in batch if r.id not in got]
        ok = list(got.values())
        if not missing:
            return ok, []
        if len(missing) == len(batch):                      # no progress at all: change the shape of the call
            return await self._split(batch, "no valid output")
        more_ok, fails = await self._extract_resilient(missing)   # recover only what is missing
        return ok + more_ok, fails

    async def _split(self, batch: list[Record], reason: str) -> tuple[list[Extraction], list[Failure]]:
        if len(batch) == 1:
            return [], [Failure(batch[0].id, batch[0].seq, reason)]
        mid = len(batch) // 2
        (a, fa), (b, fb) = await asyncio.gather(self._extract_resilient(batch[:mid]),
                                                self._extract_resilient(batch[mid:]))
        return a + b, fa + fb

    async def _vote(self, e: Extraction, rec: Record) -> Extraction:
        """Voting pattern: independent re-runs at higher temperature, majority decides."""
        async def one() -> Extraction | None:
            try:
                r = await retry_async(lambda: extract(self.gw, [rec], self.model, self.cfg.vote_temperature),
                                      attempts=self.cfg.max_attempts, base_delay=self.cfg.backoff_base_s,
                                      on_retry=self._count_retry)
                return r.get(rec.id)
            except Exception:  # noqa: BLE001
                return None
        runs = [e] + [x for x in await asyncio.gather(*(one() for _ in range(self.cfg.extra_votes))) if x]
        counts = Counter(x.sentiment for x in runs)
        top, n_top = counts.most_common(1)[0]
        tie = sum(1 for v in counts.values() if v == n_top) > 1
        e.votes = dict(counts)
        if not tie:
            winners = [x for x in runs if x.sentiment == top]
            e.sentiment = top
            e.confidence = round(sum(x.confidence for x in winners) / len(winners), 3)
            tally = Counter(t for x in runs for t in x.topics)
            e.topics = sorted(t for t, c in tally.items() if c * 2 > len(runs)) or e.topics
        return e

    async def _process_batch(self, batch: list[Record]) -> tuple[list[Extraction], list[Failure], int]:
        by_id = {r.id: r for r in batch}
        ok, fails = await self._extract_resilient(batch)
        voted = 0
        low = [e for e in ok if e.confidence < self.cfg.vote_below]
        if low and self.cfg.extra_votes > 0:
            await asyncio.gather(*(self._vote(e, by_id[e.id]) for e in low))
            voted = len(low)
        for e in ok:                                       # final routing: anything still unsure goes to a human
            c = sorted(e.votes.values(), reverse=True)
            tied = len(c) > 1 and c[0] == c[1]
            e.needs_review = e.confidence < self.cfg.vote_below or tied
        return ok, fails, voted

    # ---------- orchestration ----------
    async def run(self, input_path: str | Path, out_dir: str | Path, resume: bool = True) -> RunSummary:
        t0 = time.monotonic()
        store = Store(out_dir)
        if not resume:
            store.reset()
        done = store.done()
        summary = RunSummary(skipped_resumed=len(done))
        results: dict[str, Extraction] = dict(done)
        failures: list[Failure] = []
        queue: asyncio.Queue[list[Record] | None] = asyncio.Queue(maxsize=self.cfg.concurrency * 2)  # backpressure
        lock = asyncio.Lock()

        async def producer() -> None:
            batch: list[Record] = []
            for rec in self._read(Path(input_path), summary):
                if rec.id in done:
                    continue
                batch.append(rec)
                if len(batch) >= self.cfg.batch_size:
                    await queue.put(batch); batch = []
            if batch:
                await queue.put(batch)
            for _ in range(self.cfg.concurrency):
                await queue.put(None)

        async def worker() -> None:
            while (batch := await queue.get()) is not None:
                try:
                    ok, fails, voted = await self._process_batch(batch)
                except Exception as exc:  # noqa: BLE001  (a worker must never die and stall the queue)
                    ok, fails, voted = [], [Failure(r.id, r.seq, f"worker error: {exc}"[:200]) for r in batch], 0
                async with lock:
                    store.append(ok)                      # checkpoint immediately
                    results.update({e.id: e for e in ok})
                    failures.extend(fails)
                    summary.voted += voted

        await asyncio.gather(producer(), *(worker() for _ in range(self.cfg.concurrency)))

        items = list(results.values())
        store.write_failures(failures)
        store.finalize(items)
        summary.processed = len(items) - len(done)
        summary.failed = len(failures)
        summary.needs_review = sum(e.needs_review for e in items)
        summary.api_calls, summary.retries = self.gw.calls, self._retries
        u = self.gw.total_usage
        summary.input_tokens, summary.output_tokens = u.input_tokens, u.output_tokens
        summary.elapsed_s = time.monotonic() - t0
        if self.cfg.price_in_per_mtok is not None and self.cfg.price_out_per_mtok is not None:
            summary.estimated_cost_usd = round(u.input_tokens / 1e6 * self.cfg.price_in_per_mtok
                                               + u.output_tokens / 1e6 * self.cfg.price_out_per_mtok, 4)
        summary.aggregate = aggregate(items)
        (Path(out_dir) / "summary.json").write_text(summary.to_json(), encoding="utf-8")
        (Path(out_dir) / "report.md").write_text(to_markdown(json.loads(summary.to_json()), summary.aggregate),
                                                 encoding="utf-8")
        self.audit.write("pipeline_run", processed=summary.processed, failed=summary.failed,
                         rejected=summary.rejected, api_calls=summary.api_calls)
        return summary
