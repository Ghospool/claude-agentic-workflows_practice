"""Reduce step: roll thousands of extractions into decision-ready statistics."""
from __future__ import annotations

from collections import Counter

from .schemas import Extraction


def aggregate(items: list[Extraction]) -> dict:
    n = len(items)
    sent = Counter(e.sentiment for e in items)
    topics = Counter(t for e in items for t in e.topics)
    neg_topics = Counter(t for e in items if e.sentiment == "negative" for t in e.topics)
    return {
        "n": n,
        "sentiment": {k: sent.get(k, 0) for k in ("positive", "neutral", "negative")},
        "sentiment_pct": {k: round(100 * sent.get(k, 0) / n, 1) if n else 0.0 for k in ("positive", "neutral", "negative")},
        "topics": dict(topics.most_common()),
        "top_negative_drivers": dict(neg_topics.most_common(3)),
        "mean_confidence": round(sum(e.confidence for e in items) / n, 3) if n else 0.0,
        "records_with_pii": sum(bool(e.pii) for e in items),
    }


def to_markdown(summary_dict: dict, agg: dict) -> str:
    L = ["# Pipeline run report", "",
         f"Processed **{summary_dict['processed']}** records "
         f"({summary_dict['failed']} failed, {summary_dict['rejected']} rejected, "
         f"{summary_dict['duplicates']} duplicates, {summary_dict['needs_review']} flagged for human review).", "",
         "| Sentiment | Count | Share |", "|---|---|---|"]
    for k in ("positive", "neutral", "negative"):
        L.append(f"| {k} | {agg['sentiment'][k]} | {agg['sentiment_pct'][k]}% |")
    L += ["", "**Top negative drivers:** " + (", ".join(f"{k} ({v})" for k, v in agg["top_negative_drivers"].items()) or "none"),
          "", f"API calls: {summary_dict['api_calls']} | retries: {summary_dict['retries']} | "
              f"tokens: {summary_dict['input_tokens']} in / {summary_dict['output_tokens']} out | "
              f"elapsed: {summary_dict['elapsed_s']:.1f}s"]
    return "\n".join(L)
