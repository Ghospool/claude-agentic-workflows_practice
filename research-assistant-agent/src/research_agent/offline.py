"""Scripted heuristic policy standing in for Claude so the full agent loop runs offline.
It exercises the tool loop, budgets and verification: it is NOT a measure of real research quality."""
from __future__ import annotations

import json
import re
from typing import Any

from agentic_core import LLMResponse, MockClient, ToolCall

from .sources import tokens


def _stem(ws) -> set[str]:
    return {w[:5] for w in ws}


def _history(messages: list[dict]) -> list[tuple[str, dict, dict]]:
    uses: dict[str, tuple[str, dict]] = {}
    out = []
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in uses:
                out.append((*uses[b["tool_use_id"]], json.loads(b["content"])))
    return out


def _agent(req: dict[str, Any]) -> LLMResponse:
    msgs = req["messages"]
    plan = json.loads(re.search(r"<plan>(.*?)</plan>", str(msgs[0]["content"]), re.S)[1])["sub_questions"]
    hist = _history(msgs)
    searches = [h for h in hist if h[0] == "search"]
    reads = [h for h in hist if h[0] == "read_document" and "error" not in h[2]]
    recs = [h for h in hist if h[0] == "record_finding"]
    calls: list[ToolCall] = []
    if not searches:
        calls = [ToolCall("search", {"query": q}) for q in plan]
    elif not reads:
        ids: list[str] = []
        for _, _, res in searches:
            for r in res.get("results", [])[:2]:
                if r["doc_id"] not in ids:
                    ids.append(r["doc_id"])
        calls = [ToolCall("read_document", {"doc_id": i}) for i in ids[:4]]
    elif not recs:
        subs = [_stem(tokens(q)) for q in plan]
        for _, _, res in reads:
            sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", res["text"]) if len(s) > 25 and not s.startswith(("#", "_"))]
            scored = []
            for s in sents:
                st = _stem(tokens(s))
                best = max(range(len(subs)), key=lambda i: len(st & subs[i]))
                scored.append((len(st & subs[best]), best, s.strip()))
            for sc, idx, s in sorted(scored, key=lambda x: -x[0])[:2]:
                if sc >= 1:
                    calls.append(ToolCall("record_finding", {"sub_question": idx, "claim": s,
                                                             "source_id": res["source_id"], "quote": s}))
        if not calls:
            calls = [ToolCall("finish_research", {"coverage_summary": "no relevant evidence", "gaps": plan})]
    else:
        calls = [ToolCall("finish_research", {"coverage_summary": f"{len(recs)} findings recorded", "gaps": []})]
    return LLMResponse(tool_calls=calls)


def _handler(req: dict[str, Any]) -> LLMResponse:
    tool = req.get("force_tool")
    body = str(req["messages"][-1]["content"])
    if tool == "make_plan":
        q = re.search(r"<question>(.*?)</question>", body, re.S)[1]
        parts = [p.strip(" ?.") + "?" for p in re.split(r"\band\b|;", q) if len(p.strip()) > 12]
        return LLMResponse(tool_calls=[ToolCall(tool, {"sub_questions": parts[:4] or [q]})])
    if tool == "write_report":
        led = json.loads(re.search(r"<ledger>\n(.*)\n</ledger>", body, re.S)[1])
        sections, summary = [], []
        for i, sq in enumerate(led["sub_questions"]):
            fs = [f for f in led["findings"] if f["sub_question"] == i]
            if fs:
                sections.append({"heading": sq, "body": " ".join(f"{f['claim'].rstrip('.')} [{f['source_id']}]." for f in fs)})
                f = fs[0]; summary.append(f"{f['claim'].rstrip('.')} [{f['source_id']}].")
        return LLMResponse(tool_calls=[ToolCall(tool, {
            "title": "Research findings", "executive_summary": " ".join(summary), "sections": sections,
            "limitations": "Evidence is limited to the documents retrieved; sub-questions without findings are not answered."})])
    return _agent(req)


def offline_client() -> MockClient:
    return MockClient(_handler)
