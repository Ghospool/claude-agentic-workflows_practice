"""Lexicon-based stand-in for Claude: lets the whole pipeline run offline. Plumbing check only."""
from __future__ import annotations

import json
import re
from typing import Any

from agentic_core import LLMResponse, MockClient, ToolCall

POS = r"\b(good|great|excellent|fast|helpful|friendly|easy|intuitive|fair|love|loved|early|dependable|solid)\b"
NEG = (r"\b(expensive|poor|cheap|late|damaged|useless|confusing|hard|crashing|crashes|outages?|broken|"
       r"never replied|awful|terrible)\b")
NEU = r"\b(okay|average|nothing special|as described)\b"
TOPIC_PATTERNS = {
    "pricing": r"\b(pricing|price|expensive|value)\b", "quality": r"\b(quality|cheap|build)\b",
    "delivery": r"\b(delivery|package|arrived)\b", "support": r"\b(support|replied)\b",
    "usability": r"\b(easy|interface|intuitive|use)\b", "reliability": r"\b(crash\w*|outages?|dependable|solid)\b",
}


def analyse(text: str) -> tuple[str, list[str], float]:
    t = text.lower()
    p, n, u = (len(re.findall(x, t)) for x in (POS, NEG, NEU))
    if p and n:
        sent, conf = "neutral", 0.5          # mixed signals (sarcasm, negation): honestly unsure
    elif p:
        sent, conf = "positive", 0.9
    elif n:
        sent, conf = "negative", 0.9
    else:
        sent, conf = "neutral", 0.85 if u else 0.6
    topics = [k for k, pat in TOPIC_PATTERNS.items() if re.search(pat, t)] or ["other"]
    return sent, topics, conf


def _handler(req: dict[str, Any]) -> LLMResponse:
    body = str(req["messages"][-1]["content"])
    records = json.loads(re.search(r"<records>\n(.*)\n</records>", body, re.S)[1])
    results = []
    for r in records:
        s, tps, c = analyse(r["text"])
        results.append({"id": r["id"], "sentiment": s, "topics": tps, "confidence": c,
                        "summary": r["text"][:60]})
    return LLMResponse(tool_calls=[ToolCall("extract_batch", {"results": results})])


def offline_client() -> MockClient:
    return MockClient(_handler)
