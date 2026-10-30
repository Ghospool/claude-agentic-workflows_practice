"""Score the pipeline against labelled data: quality, trust (auto-accepted accuracy) and efficiency."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentic_core.evals import classification_report

from .pipeline import DataPipeline


async def run_eval(pipe: DataPipeline, data_dir: str | Path) -> dict:
    d = Path(data_dir)
    labels = {j["id"]: j for j in map(json.loads, (d / "labels.jsonl").read_text(encoding="utf-8").splitlines())}
    with tempfile.TemporaryDirectory() as out:
        s = await pipe.run(d / "reviews.jsonl", out, resume=False)
        got = [json.loads(l) for l in (Path(out) / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [g["id"] for g in got]
    y_true = [labels[i]["sentiment"] for i in ids]
    y_pred = [g["sentiment"] for g in got]
    rep = classification_report(y_true, y_pred)
    auto = [(t, p) for t, p, g in zip(y_true, y_pred, got) if not g["needs_review"]]
    tp = fp = fn = 0
    for g in got:
        exp, pr = set(labels[g["id"]]["topics"]), set(g["topics"])
        tp += len(exp & pr); fp += len(pr - exp); fn += len(exp - pr)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "records": len(got), "sentiment_accuracy": rep.accuracy,
        "sentiment_macro_f1": sum(m["f1"] for m in rep.per_class.values()) / len(rep.per_class),
        "auto_accepted_accuracy": sum(t == p for t, p in auto) / max(1, len(auto)),
        "human_review_rate": sum(g["needs_review"] for g in got) / max(1, len(got)),
        "topic_precision": prec, "topic_recall": rec, "topic_f1": 2 * prec * rec / max(1e-9, prec + rec),
        "failed": s.failed, "api_calls": s.api_calls, "records_per_call": len(got) / max(1, s.api_calls),
        "input_tokens": s.input_tokens, "output_tokens": s.output_tokens,
    }
