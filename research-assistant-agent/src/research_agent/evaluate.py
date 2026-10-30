"""Evaluate on questions with expected facts: coverage, citation integrity, safety and cost."""
from __future__ import annotations

import json
from pathlib import Path

from .agent import ResearchAgent
from .sources import LocalCorpusProvider


async def run_eval(client, data_dir: str | Path, settings=None) -> dict:
    d = Path(data_dir)
    qs = [json.loads(l) for l in (d / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    provider = LocalCorpusProvider(d / "corpus")
    tot = {"facts": 0, "hit": 0, "invalid": 0, "uncited": 0, "leak": 0, "steps": 0, "calls": 0, "rej": 0,
           "in": 0, "out": 0, "early": 0}
    for q in qs:
        res = await ResearchAgent(client, provider, settings=settings).run(q["question"])
        text = res.report_markdown.lower()
        tot["facts"] += len(q["expected_facts"])
        tot["hit"] += sum(f.lower() in text for f in q["expected_facts"])
        tot["leak"] += any(c.lower() in text for c in q.get("forbidden", []))
        tot["invalid"] += res.invalid_citations; tot["uncited"] += res.uncited_sentences
        tot["steps"] += res.steps; tot["calls"] += res.tool_calls; tot["rej"] += res.rejected_findings
        tot["in"] += res.input_tokens; tot["out"] += res.output_tokens; tot["early"] += res.stop_reason != "finished"
    n = len(qs)
    return {"questions": n, "fact_recall": tot["hit"] / max(1, tot["facts"]), "invalid_citations": tot["invalid"],
            "uncited_sentences": tot["uncited"], "injection_leaks": tot["leak"], "early_stops": tot["early"],
            "avg_steps": tot["steps"] / n, "avg_tool_calls": tot["calls"] / n,
            "rejected_findings": tot["rej"], "input_tokens": tot["in"], "output_tokens": tot["out"]}
