import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agentic_core import LLMResponse, MockClient, ToolCall, TransientError
from data_pipeline import DataPipeline, PipelineConfig
from data_pipeline.evaluate import run_eval
from data_pipeline.offline import _handler as lexicon_handler
from data_pipeline.offline import offline_client
from data_pipeline.ratelimit import TokenBucket
from data_pipeline.retry import retry_async

DATA = Path(__file__).resolve().parents[1] / "data"
FAST = dict(backoff_base_s=0.001, requests_per_second=10_000)


def write(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def recs(n):
    return [{"id": f"r{i}", "text": "Excellent build quality" if i % 2 else "Way too expensive"} for i in range(n)]


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


class Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name); self.inp = self.dir / "in.jsonl"; self.out = self.dir / "out"

    def pipe(self, client=None, **cfg):
        return DataPipeline(client or offline_client(), PipelineConfig(**{**FAST, **cfg}))


class IngestTests(Base):
    async def test_bad_lines_duplicates_and_pii(self):
        write(self.inp, [{"id": "a", "text": "good, mail me bob@x.com"}, "not json", {"id": "b"}, {"id": "a", "text": "dup"},
                         {"id": "c", "text": "   "}, {"id": "d", "text": "terrible"}])
        c = offline_client()
        s = await self.pipe(c).run(self.inp, self.out)
        self.assertEqual((s.processed, s.rejected, s.duplicates), (2, 3, 1))
        self.assertNotIn("bob@x.com", " ".join(str(x["messages"]) for x in c.calls))
        self.assertEqual(read(self.out / "results.jsonl")[0]["pii"], {"EMAIL": 1})


class ParallelTests(Base):
    async def test_output_preserves_input_order_and_batches(self):
        write(self.inp, recs(95))
        c = offline_client()
        s = await self.pipe(c, batch_size=10, concurrency=4).run(self.inp, self.out)
        self.assertEqual([r["id"] for r in read(self.out / "results.jsonl")], [f"r{i}" for i in range(95)])
        self.assertEqual(s.api_calls, 10)

    async def test_concurrency_never_exceeds_limit(self):
        class Slow(MockClient):
            async def complete(self, **kw):
                await asyncio.sleep(0.01)
                return await super().complete(**kw)
        write(self.inp, recs(100))
        p = self.pipe(Slow(lexicon_handler), batch_size=5, concurrency=3)
        await p.run(self.inp, self.out)
        self.assertLessEqual(p.gw.max_in_flight, 3)
        self.assertGreaterEqual(p.gw.max_in_flight, 2)       # and it really did run in parallel

    async def test_token_bucket_limits_rate(self):
        b = TokenBucket(rate_per_s=50, capacity=1)
        loop = asyncio.get_running_loop(); t0 = loop.time()
        for _ in range(6):
            await b.acquire()
        self.assertGreaterEqual(loop.time() - t0, 0.08)      # 5 refills at 50/s ~ 0.1s


class ResilienceTests(Base):
    async def test_transient_errors_are_retried(self):
        state = {"n": 0}
        def h(req):
            state["n"] += 1
            if state["n"] <= 2:
                raise TransientError("429")
            return lexicon_handler(req)
        write(self.inp, recs(10))
        s = await self.pipe(MockClient(h), batch_size=10).run(self.inp, self.out)
        self.assertEqual((s.processed, s.failed, s.retries), (10, 0, 2))

    async def test_poison_record_isolated_to_dlq(self):
        def h(req):
            if "POISON" in str(req["messages"][-1]["content"]):
                raise ValueError("model refused")
            return lexicon_handler(req)
        rows = recs(16); rows[5]["text"] = "POISON"
        write(self.inp, rows)
        s = await self.pipe(MockClient(h), batch_size=16).run(self.inp, self.out)
        self.assertEqual((s.processed, s.failed), (15, 1))
        self.assertEqual(read(self.out / "failures.jsonl")[0]["id"], "r5")

    async def test_partial_output_recovers_missing_records(self):
        def h(req):
            resp = lexicon_handler(req)
            items = resp.tool_calls[0].input["results"]
            if len(items) > 1:                                   # model drops last record + invents one
                resp.tool_calls[0].input["results"] = items[:-1] + [{"id": "ghost", "sentiment": "positive",
                    "topics": [], "summary": "", "confidence": 1}]
            return resp
        write(self.inp, recs(8))
        s = await self.pipe(MockClient(h), batch_size=8).run(self.inp, self.out)
        self.assertEqual((s.processed, s.failed), (8, 0))
        self.assertNotIn("ghost", [r["id"] for r in read(self.out / "results.jsonl")])

    async def test_resume_skips_completed_work(self):
        write(self.inp, recs(30))
        await self.pipe(batch_size=10).run(self.inp, self.out)
        c = offline_client()
        s = await self.pipe(c, batch_size=10).run(self.inp, self.out)
        self.assertEqual((s.skipped_resumed, s.processed, len(c.calls)), (30, 0, 0))
        write(self.inp, recs(40))                                 # 10 new records arrive
        c2 = offline_client()
        s2 = await self.pipe(c2, batch_size=10).run(self.inp, self.out)
        self.assertEqual((s2.processed, len(c2.calls)), (10, 1))
        self.assertEqual(len(read(self.out / "results.jsonl")), 40)

    async def test_torn_checkpoint_line_is_tolerated(self):
        write(self.inp, recs(5))
        await self.pipe().run(self.inp, self.out)
        with (self.out / "results.partial.jsonl").open("a") as f:
            f.write('{"id": "r9", "seq"')                         # simulated crash mid-write
        s = await self.pipe().run(self.inp, self.out)
        self.assertEqual(s.failed, 0)


class VotingTests(Base):
    async def test_majority_overrides_unsure_first_answer(self):
        calls = {"n": 0}
        def h(req):
            calls["n"] += 1
            sent, conf = ("neutral", 0.4) if req.get("temperature", 0) == 0 else ("negative", 0.9)
            rid = json.loads(str(req["messages"][-1]["content"]).split("\n", 1)[1].rsplit("\n", 1)[0])[0]["id"]
            return LLMResponse(tool_calls=[ToolCall("extract_batch", {"results": [
                {"id": rid, "sentiment": sent, "topics": ["support"], "summary": "s", "confidence": conf}]})])
        write(self.inp, [{"id": "x", "text": "ambiguous"}])
        s = await self.pipe(MockClient(h)).run(self.inp, self.out)
        r = read(self.out / "results.jsonl")[0]
        self.assertEqual((r["sentiment"], s.voted, calls["n"]), ("negative", 1, 3))
        self.assertEqual(r["votes"], {"neutral": 1, "negative": 2})
        self.assertFalse(r["needs_review"])

    async def test_unresolved_vote_goes_to_human(self):
        def h(req):
            return LLMResponse(tool_calls=[ToolCall("extract_batch", {"results": [
                {"id": "x", "sentiment": "neutral", "topics": ["other"], "summary": "", "confidence": 0.3}]})])
        write(self.inp, [{"id": "x", "text": "???"}])
        await self.pipe(MockClient(h)).run(self.inp, self.out)
        self.assertTrue(read(self.out / "results.jsonl")[0]["needs_review"])


class RetryUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_transient_not_retried(self):
        n = {"c": 0}
        async def f():
            n["c"] += 1; raise ValueError("bad")
        with self.assertRaises(ValueError):
            await retry_async(f, attempts=3, base_delay=0.001)
        self.assertEqual(n["c"], 1)


class EvalTests(Base):
    async def test_golden_eval(self):
        r = await run_eval(self.pipe(batch_size=20, concurrency=6), DATA)
        self.assertGreaterEqual(r["sentiment_accuracy"], 0.9)
        self.assertGreaterEqual(r["auto_accepted_accuracy"], r["sentiment_accuracy"])
        self.assertEqual(r["failed"], 0)
        self.assertGreater(r["records_per_call"], 5)


if __name__ == "__main__":
    unittest.main()
