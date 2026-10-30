"""Keyword-based stand-in for Claude so the full pipeline runs and tests offline.
It is NOT a quality proxy: run the eval with a real key to measure actual accuracy."""
from __future__ import annotations

from typing import Any

from agentic_core import LLMResponse, MockClient, ToolCall

from .classifier import TOOL_NAME

KEYWORDS = {
    "escalation": ["lawyer", "legal action", "sue", "gdpr", "breach", "hacked", "regulator",
                   "harass", "threat", "journalist", "unsafe"],
    "billing": ["charge", "invoice", "refund", "billed", "payment", "receipt", "credit card"],
    "technical": ["error", "crash", "bug", "api", "timeout", "500", "not loading", "slow", "outage"],
    "account": ["password", "login", "log in", "locked", "2fa", "access", "delete my", "export my"],
    "sales": ["pricing", "quote", "upgrade", "demo", "enterprise", "seats", "purchase"],
}


def _handler(req: dict[str, Any]) -> LLMResponse:
    text = str(req["messages"][-1]["content"]).lower()
    if req.get("force_tool") == TOOL_NAME:
        scores = {r: sum(k in text for k in ks) for r, ks in KEYWORDS.items()}
        route, hits = max(scores.items(), key=lambda kv: kv[1])
        if hits == 0:
            route, conf = "general", 0.75
        else:
            conf = min(0.95, 0.6 + 0.15 * hits)
        urgency = "high" if route == "escalation" else "medium"
        return LLMResponse(tool_calls=[ToolCall(TOOL_NAME, {
            "route": route, "confidence": conf, "urgency": urgency, "reasoning": "keyword match"})])
    return LLMResponse(text="Thanks for reaching out. We're looking into this and will follow up "
                            "with next steps shortly.")


def offline_client() -> MockClient:
    return MockClient(_handler)
