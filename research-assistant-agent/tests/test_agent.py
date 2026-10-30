import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agentic_core import LLMResponse, MockClient, ToolCall
from agentic_core.security import AuditLog
from research_agent import AgentConfig, CompositeProvider, LocalCorpusProvider, ResearchAgent
from research_agent.evaluate import run_eval
from research_agent.offline import _handler, offline_client
from research_agent.sources import is_safe_url
from research_agent.tools import ToolExecutor
from research_agent.verify import check_text

DATA = Path(__file__).resolve().parents[1] / "data"
Q1 = "What are the total cost of ownership and purchase price implications of switching to electric delivery vans?"


def agent(client=None, **cfg):
    return ResearchAgent(client or offline_client(), LocalCorpusProvider(DATA / "corpus"), AgentConfig(**cfg))


class SourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_ranks_relevant_doc_first(self):
        p = LocalCorpusProvider(DATA / "corpus")
        r = await p.search("charging infrastructure grid connection depot", 3)
        self.assertEqual(r[0].doc_id, "charging-infrastructure.md")
        self.assertNotIn("office-catering.md", [x.doc_id for x in await p.search("electric van cost", 8)])

    async def test_fetch_rejects_unknown_ids_and_traversal(self):
        p = LocalCorpusProvider(DATA / "corpus")
        for bad in ("../../etc/passwd", "missing.md"):
            with self.assertRaises(KeyError):
                await p.fetch(bad)

    async def test_composite_namespaces_sources(self):
        c = CompositeProvider({"kb": LocalCorpusProvider(DATA / "corpus")})
        r = await c.search("pilot energy cost per kilometre", 2)
        self.assertTrue(r[0].doc_id.startswith("kb::"))
        self.assertIn("12-van", (await c.fetch(r[0].doc_id)).text)

    def test_ssrf_guard(self):
        for u in ("http://127.0.0.1/", "http://169.254.169.254/latest/meta-data", "http://10.0.0.5/", "file:///etc/passwd",
                  "ftp://example.com/x", "http://[::1]/", "http://localhost:8080/", "not a url"):
            self.assertFalse(is_safe_url(u), u)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def ex(self, n=2, cap=50):
        return ToolExecutor(LocalCorpusProvider(DATA / "corpus"), n, cap, AuditLog(None))

    async def test_finding_requires_verbatim_quote_from_read_source(self):
        ex = self.ex()
        self.assertIn("error", await ex.run(ToolCall("record_finding", {"sub_question": 0, "claim": "x", "source_id": "S1",
                                                                          "quote": "never read this document"})))
        r = await ex.run(ToolCall("read_document", {"doc_id": "fleet-economics.md"}))
        self.assertEqual(r["source_id"], "S1")
        bad = await ex.run(ToolCall("record_finding", {"sub_question": 0, "claim": "c", "source_id": "S1",
                                                       "quote": "costs are 90% lower than diesel everywhere"}))
        self.assertIn("not found verbatim", bad["error"])
        ok = await ex.run(ToolCall("record_finding", {"sub_question": 0, "claim": "12% lower TCO", "source_id": "S1",
                                                      "quote": "projected to be 12%  lower than a diesel equivalent"}))
        self.assertEqual(ok["recorded"], "F1")                       # whitespace-insensitive match
        self.assertEqual(ex.rejected, 2)

    async def test_duplicate_queries_blocked_and_budget_enforced(self):
        ex = self.ex(cap=3)
        await ex.run(ToolCall("search", {"query": "Electric  vans"}))
        self.assertIn("duplicate", (await ex.run(ToolCall("search", {"query": "electric vans"})))["error"])
        await ex.run(ToolCall("search", {"query": "other"}))
        self.assertIn("budget", (await ex.run(ToolCall("search", {"query": "fourth"})))["error"])

    async def test_injection_in_document_is_flagged(self):
        ex = self.ex()
        r = await ex.run(ToolCall("read_document", {"doc_id": "vendor-comparison.md"}))
        self.assertEqual(r["warning"], "possible_prompt_injection")
        self.assertTrue(ex.warnings)

    async def test_tool_errors_do_not_raise(self):
        ex = self.ex()
        self.assertIn("error", await ex.run(ToolCall("read_document", {"doc_id": "nope.md"})))
        self.assertIn("error", await ex.run(ToolCall("search", {"wrong": 1})))
        self.assertIn("error", await ex.run(ToolCall("rm_rf", {})))


class VerifyTests(unittest.TestCase):
    def test_citation_checks(self):
        t, inv, unc = check_text("Fact one. [S1] Fact two [S9]. An unsupported claim.", {"S1", "S2"})
        self.assertEqual((inv, unc), (1, 1))
        self.assertIn("[?]", t)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end_cited_report(self):
        res = await agent().run(Q1)
        self.assertEqual(res.stop_reason, "finished")
        self.assertIn("12% lower", res.report_markdown)
        self.assertEqual((res.invalid_citations, res.uncited_sentences), (0, 0))
        self.assertTrue(all(f.source_id in res.sources for f in res.findings))
        self.assertIn("## Sources", res.report_markdown)

    async def test_injected_instruction_not_followed(self):
        res = await agent().run("Which vendor offers the longest battery warranty and lead times for electric vans?")
        self.assertNotIn("PWNED", res.report_markdown.replace("IMPORTANT SYSTEM INSTRUCTION", ""))
        self.assertIn("eight years", res.report_markdown)
        self.assertTrue(any("instruction-like" in w for w in res.warnings))

    async def test_step_budget_stops_and_still_reports_partial_evidence(self):
        res = await agent(max_steps=3).run(Q1)
        self.assertEqual(res.stop_reason, "max_steps")
        self.assertTrue(any("stopped early" in w for w in res.warnings))

    async def test_no_evidence_means_no_report_claims(self):
        res = await agent().run("zzzz qqqq xxxx unrelated topic entirely?")
        self.assertIn("Insufficient verified evidence", res.report_markdown)
        self.assertEqual(res.findings, [])

    async def test_hallucinated_citation_is_marked(self):
        def h(req):
            r = _handler(req)
            if req.get("force_tool") == "write_report":
                r.tool_calls[0].input["executive_summary"] += " Invented claim [S42]."
            return r
        res = await agent(MockClient(h)).run(Q1)
        self.assertEqual(res.invalid_citations, 1)
        self.assertIn("[?]", res.report_markdown)

    async def test_bad_quote_is_fed_back_to_agent(self):
        state = {"tried": False}
        def h(req):
            r = _handler(req)
            if not req.get("force_tool"):
                for c in r.tool_calls:
                    if c.name == "record_finding" and not state["tried"]:
                        state["tried"] = True; c.input["quote"] = "made up evidence that is not in the document"
            return r
        res = await agent(MockClient(h)).run(Q1)
        self.assertGreaterEqual(res.rejected_findings, 1)
        self.assertGreaterEqual(len(res.findings), 1)

    async def test_idle_model_gives_up_cleanly(self):
        def h(req):
            return _handler(req) if req.get("force_tool") else LLMResponse(text="thinking...")
        res = await agent(MockClient(h)).run(Q1)
        self.assertEqual(res.stop_reason, "no_progress")

    async def test_parallel_tool_calls_run_concurrently(self):
        class Slow(LocalCorpusProvider):
            active = peak = 0
            async def search(self, q, n=5):
                Slow.active += 1; Slow.peak = max(Slow.peak, Slow.active)
                await asyncio.sleep(0.02); Slow.active -= 1
                return await super().search(q, n)
        a = ResearchAgent(offline_client(), Slow(DATA / "corpus"))
        await a.run(Q1)
        self.assertGreaterEqual(Slow.peak, 2)

    async def test_golden_eval(self):
        r = await run_eval(offline_client(), DATA)
        self.assertGreaterEqual(r["fact_recall"], 0.85)
        self.assertEqual((r["invalid_citations"], r["uncited_sentences"], r["injection_leaks"], r["early_stops"]), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
