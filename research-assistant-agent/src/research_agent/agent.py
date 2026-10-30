"""Autonomous research agent.

PLAN (1 forced call) -> RESEARCH LOOP (model chooses tools; parallel tool calls; hard budgets)
-> SYNTHESISE from the evidence ledger only (1 forced call) -> VERIFY citations deterministically.

The model never writes from memory: the report may only use recorded findings, each backed by a
verbatim quote that the executor has checked against the source text."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from importlib import resources

from agentic_core import LLMClient, Settings, ToolCall, get_settings
from agentic_core.security import AuditLog

from .schemas import ResearchResult
from .sources import SourceProvider
from .tools import TOOLS, ToolExecutor
from .verify import check_text

log = logging.getLogger(__name__)

PLAN_TOOL = {"name": "make_plan", "description": "Break the question into focused sub-questions.",
             "input_schema": {"type": "object", "required": ["sub_questions"], "properties": {
                 "sub_questions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}}}}
REPORT_TOOL = {"name": "write_report", "description": "Write the final report using only the evidence ledger.",
               "input_schema": {"type": "object", "required": ["title", "executive_summary", "sections", "limitations"],
                                "properties": {
                                    "title": {"type": "string"}, "executive_summary": {"type": "string"},
                                    "sections": {"type": "array", "items": {"type": "object",
                                        "required": ["heading", "body"],
                                        "properties": {"heading": {"type": "string"}, "body": {"type": "string"}}}},
                                    "conflicts": {"type": "array", "items": {"type": "string"}},
                                    "limitations": {"type": "string"}}}}


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_tool_calls: int = 30
    max_input_tokens: int = 300_000      # cumulative budget across the run
    max_idle_turns: int = 2              # consecutive turns with no tool call before giving up


def _prompt(name: str) -> str:
    return resources.files("research_agent.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


class ResearchAgent:
    def __init__(self, client: LLMClient, provider: SourceProvider, config: AgentConfig | None = None,
                 settings: Settings | None = None) -> None:
        self.client, self.provider = client, provider
        self.cfg = config or AgentConfig()
        self.s = settings or get_settings()
        self.audit = AuditLog(self.s.audit_log_path)

    async def _forced(self, tool: dict, system: str, user: str, max_tokens: int) -> dict:
        r = await self.client.complete(system=system, messages=[{"role": "user", "content": user}],
                                       model=self.s.smart_model, max_tokens=max_tokens, tools=[tool],
                                       force_tool=tool["name"])
        for c in r.tool_calls:
            if c.name == tool["name"]:
                return c.input
        raise ValueError(f"model returned no {tool['name']} call")

    async def run(self, question: str) -> ResearchResult:
        u0 = (self.client.total_usage.input_tokens, self.client.total_usage.output_tokens)
        try:
            plan = await self._forced(PLAN_TOOL, _prompt("plan"), f"<question>{question}</question>", 500)
            subs = [str(s) for s in plan.get("sub_questions", []) if str(s).strip()][:5] or [question]
        except ValueError:
            subs = [question]                                   # planning is optional: degrade, don't fail

        ex = ToolExecutor(self.provider, len(subs), self.cfg.max_tool_calls, self.audit)
        user0 = (f"<question>{question}</question>\n<plan>{json.dumps({'sub_questions': subs})}</plan>\n"
                 "Begin the research.")
        messages: list[dict] = [{"role": "user", "content": user0}]
        stop, steps, idle = "finished", 0, 0

        while not ex.finished:
            if steps >= self.cfg.max_steps:
                stop = "max_steps"; break
            if self.client.total_usage.input_tokens - u0[0] > self.cfg.max_input_tokens:
                stop = "token_budget"; break
            if ex.calls >= self.cfg.max_tool_calls:
                stop = "tool_budget"; break
            steps += 1
            resp = await self.client.complete(system=_prompt("research"), messages=messages, model=self.s.smart_model,
                                              max_tokens=2000, tools=TOOLS)
            calls = [ToolCall(c.name, c.input, c.id or f"toolu_{steps}_{i}") for i, c in enumerate(resp.tool_calls)]
            blocks = ([{"type": "text", "text": resp.text}] if resp.text else []) + \
                     [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
            messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no output)"}]})
            if not calls:
                idle += 1
                if idle > self.cfg.max_idle_turns:
                    stop = "no_progress"; break
                messages.append({"role": "user", "content": "Continue with tools, or call finish_research."})
                continue
            idle = 0
            results = await asyncio.gather(*(ex.run(c) for c in calls))     # parallel tool execution
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(r, ensure_ascii=False),
                 "is_error": "error" in r} for c, r in zip(calls, results)]})

        warnings = list(ex.warnings)
        if stop != "finished":
            warnings.append(f"research stopped early ({stop}); coverage may be incomplete")
        if ex.gaps:
            warnings.append("gaps reported by agent: " + "; ".join(ex.gaps))

        report, invalid, uncited = await self._synthesise(question, subs, ex, warnings)
        res = ResearchResult(
            question=question, sub_questions=subs, report_markdown=report, findings=ex.findings,
            sources={k: {"doc_id": s.doc_id, "title": s.title, "url": s.url} for k, s in ex.sources.items()},
            warnings=warnings, stop_reason=stop, steps=steps, tool_calls=ex.calls, rejected_findings=ex.rejected,
            invalid_citations=invalid, uncited_sentences=uncited,
            input_tokens=self.client.total_usage.input_tokens - u0[0],
            output_tokens=self.client.total_usage.output_tokens - u0[1])
        self.audit.write("research_run", question=question[:120], **res.metrics())
        return res

    async def _synthesise(self, question: str, subs: list[str], ex: ToolExecutor, warnings: list[str]):
        if not ex.findings:                                 # never write a report without evidence
            warnings.append("no verified evidence was found")
            return (f"# Research report\n\n**Question:** {question}\n\nInsufficient verified evidence was found "
                    "in the available sources to answer this question. No claims are made."), 0, 0
        ledger = {"sub_questions": subs,
                  "findings": [{"id": f.id, "sub_question": f.sub_question, "claim": f.claim,
                                "source_id": f.source_id, "quote": f.quote} for f in ex.findings],
                  "sources": {k: s.title for k, s in ex.sources.items()}}
        draft = await self._forced(REPORT_TOOL, _prompt("synthesize"),
                                   f"<question>{question}</question>\n<ledger>\n{json.dumps(ledger)}\n</ledger>", 3500)
        valid = {f.source_id for f in ex.findings}
        invalid = uncited = 0
        parts: list[str] = []
        def clean(t: str) -> str:
            nonlocal invalid, uncited
            c, i, u = check_text(t, valid)
            invalid += i; uncited += u
            return c
        title = str(draft.get("title", "Research report"))
        summary = clean(str(draft.get("executive_summary", "")))
        parts += [f"# {title}", "", f"**Question:** {question}", "", "## Executive summary", "", summary, ""]
        cited: set[str] = set(m for m in re.findall(r"\[(S\d+)\]", summary))
        for sec in draft.get("sections", []):
            body = clean(str(sec.get("body", "")))
            cited |= set(re.findall(r"\[(S\d+)\]", body))
            parts += [f"## {sec.get('heading', '')}", "", body, ""]
        if draft.get("conflicts"):
            parts += ["## Conflicting evidence", ""] + [f"- {clean(str(c))}" for c in draft["conflicts"]] + [""]
        parts += ["## Limitations", "", str(draft.get("limitations", "")), ""]
        if invalid:
            warnings.append(f"{invalid} citation(s) did not match a source and were marked [?]")
        if uncited:
            warnings.append(f"{uncited} sentence(s) in the report carry no citation; treat with caution")
        parts += ["## Sources", ""] + [f"- [{k}] {ex.sources[k].title} ({ex.sources[k].url or ex.sources[k].doc_id})"
                                       for k in sorted(cited, key=lambda x: int(x[1:])) if k in ex.sources]
        if warnings:
            parts += ["", "## Reviewer notes", ""] + [f"- {w}" for w in warnings]
        return "\n".join(parts), invalid, uncited
