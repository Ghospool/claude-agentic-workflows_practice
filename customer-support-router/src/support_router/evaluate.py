"""Run the golden dataset through the router and report quality + governance metrics."""
from __future__ import annotations

import json
from pathlib import Path

from agentic_core.evals import ClassificationReport, classification_report

from .pipeline import SupportRouter
from .schemas import RouterResult, Ticket


def load_dataset(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


async def run_eval(router: SupportRouter, path: str | Path) -> tuple[ClassificationReport, dict]:
    rows = load_dataset(path)
    results: list[RouterResult] = await router.handle_many(
        [Ticket(id=r["id"], text=r["text"]) for r in rows])
    report = classification_report([r["expected"] for r in rows], [x.route.value for x in results])
    n = len(rows)
    esc_expected = [r["expected"] == "escalation" for r in rows]
    caught = sum(e and x.needs_human for e, x in zip(esc_expected, results))
    auto = [x for x in results if not x.needs_human]
    auto_correct = sum(x.route.value == r["expected"] for x, r in zip(results, rows) if not x.needs_human)
    extra = {
        "human_handoff_rate": sum(x.needs_human for x in results) / n,
        "escalation_recall": caught / max(1, sum(esc_expected)),   # safety-critical: target 100%
        "automation_precision": auto_correct / max(1, len(auto)),  # of auto-replied, share correctly routed
        "pii_tickets": sum(bool(x.pii_redacted) for x in results),
        "total_input_tokens": sum(x.input_tokens for x in results),
        "total_output_tokens": sum(x.output_tokens for x in results),
    }
    return report, extra
