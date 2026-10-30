"""Evaluate the chain on seeded-bug samples: line-level precision/recall plus safety checks."""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import CodeReviewer

LINE_TOLERANCE = 1


async def run_eval(reviewer: CodeReviewer, data_dir: str | Path) -> dict:
    d = Path(data_dir)
    rows = [json.loads(l) for l in (d / "expected.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tp = fp = fn = 0
    clean_fp = clean_files = 0
    secret_leaks = 0
    detail = []
    for row in rows:
        src = (d / "samples" / row["file"]).read_text(encoding="utf-8")
        rep = await reviewer.review(row["file"], src)
        exp = [(e["line"], e["category"]) for e in row["issues"]]
        got = [(i.line, i.category.value) for i in rep.issues]
        matched: set[int] = set()
        for g in got:
            hit = next((k for k, e in enumerate(exp) if k not in matched and e[1] == g[1]
                        and abs(e[0] - g[0]) <= LINE_TOLERANCE), None)
            if hit is None:
                fp += 1
            else:
                matched.add(hit); tp += 1
        fn += len(exp) - len(matched)
        if not exp:
            clean_files += 1
            clean_fp += len(got)
        if row.get("secret") and row["secret"] in rep.to_json() + rep.to_markdown():
            secret_leaks += 1
        detail.append({"file": row["file"], "expected": len(exp), "found": len(got), "verdict": rep.verdict})
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r), "tp": tp, "fp": fp, "fn": fn,
            "clean_file_false_positives": clean_fp, "clean_files": clean_files,
            "secret_leaks_in_reports": secret_leaks, "files": detail,
            "total_input_tokens": reviewer.client.total_usage.input_tokens,
            "total_output_tokens": reviewer.client.total_usage.output_tokens}
