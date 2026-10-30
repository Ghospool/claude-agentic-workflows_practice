"""The individual links of the chain. Each step is a pure async function: easy to test and swap."""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

from agentic_core import LLMClient

from .schemas import Issue

ISSUE_PROPS = {
    "line": {"type": "integer"},
    "category": {"type": "string", "enum": ["security", "bug", "reliability", "performance", "maintainability"]},
    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
    "title": {"type": "string"},
    "explanation": {"type": "string"},
    "evidence": {"type": "string", "description": "Verbatim code of the flagged line."},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
}
TOOLS = {
    "report_issues": {
        "name": "report_issues", "description": "Report issues found in the code.",
        "input_schema": {"type": "object", "required": ["issues"], "properties": {"issues": {
            "type": "array", "items": {"type": "object", "properties": ISSUE_PROPS,
                                       "required": list(ISSUE_PROPS)}}}}},
    "verify_issues": {
        "name": "verify_issues", "description": "Give a verdict for each candidate issue.",
        "input_schema": {"type": "object", "required": ["verdicts"], "properties": {"verdicts": {
            "type": "array", "items": {"type": "object", "required": ["id", "valid", "reason"], "properties": {
                "id": {"type": "string"}, "valid": {"type": "boolean"}, "reason": {"type": "string"},
                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]}}}}}}},
    "suggest_fixes": {
        "name": "suggest_fixes", "description": "Propose a minimal fix for each confirmed issue.",
        "input_schema": {"type": "object", "required": ["fixes"], "properties": {"fixes": {
            "type": "array", "items": {"type": "object", "required": ["id", "fix_description", "patch"], "properties": {
                "id": {"type": "string"}, "fix_description": {"type": "string"}, "patch": {"type": "string"}}}}}}},
}


def prompt(name: str) -> str:
    return resources.files("code_review.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def numbered(source: str) -> str:
    return "\n".join(f"{n:>4}| {l}" for n, l in enumerate(source.splitlines(), 1))


async def _call(client: LLMClient, tool: str, system: str, user: str, model: str, max_tokens: int) -> dict[str, Any]:
    r = await client.complete(system=system, messages=[{"role": "user", "content": user}], model=model,
                              max_tokens=max_tokens, tools=[TOOLS[tool]], force_tool=tool)
    for c in r.tool_calls:
        if c.name == tool:
            return c.input
    raise ValueError(f"model returned no {tool} call")


async def analyze(client: LLMClient, filename: str, source: str, model: str) -> list[Issue]:
    out = await _call(client, "report_issues", prompt("analyze"),
                      f"<file name=\"{filename}\">\n{numbered(source)}\n</file>", model, 3000)
    issues = []
    for raw in out.get("issues", []):
        try:
            issues.append(Issue(**raw))
        except (TypeError, ValueError):
            continue  # malformed candidate: skip rather than crash the review
    return issues


async def verify(client: LLMClient, filename: str, source: str, issues: list[Issue], model: str) -> dict[str, dict]:
    cands = json.dumps([{"id": i.id, "line": i.line, "category": i.category.value, "severity": i.severity.value,
                         "title": i.title, "explanation": i.explanation, "evidence": i.evidence} for i in issues])
    out = await _call(client, "verify_issues", prompt("verify"),
                      f"<file name=\"{filename}\">\n{numbered(source)}\n</file>\n<candidates>\n{cands}\n</candidates>",
                      model, 2000)
    return {v["id"]: v for v in out.get("verdicts", []) if "id" in v}


async def suggest(client: LLMClient, filename: str, source: str, issues: list[Issue], model: str) -> dict[str, dict]:
    cands = json.dumps([{"id": i.id, "line": i.line, "title": i.title, "explanation": i.explanation} for i in issues])
    out = await _call(client, "suggest_fixes", prompt("suggest"),
                      f"<file name=\"{filename}\">\n{numbered(source)}\n</file>\n<confirmed_issues>\n{cands}\n</confirmed_issues>",
                      model, 3000)
    return {f["id"]: f for f in out.get("fixes", []) if "id" in f}


async def summarize(client: LLMClient, filename: str, issues: list[Issue], model: str) -> str:
    facts = json.dumps([{"line": i.line, "severity": i.severity.value, "title": i.title} for i in issues])
    r = await client.complete(system=prompt("summarize"), model=model, max_tokens=300,
                              messages=[{"role": "user", "content": f"File: {filename}\nConfirmed issues: {facts}"}])
    return r.text.strip()
