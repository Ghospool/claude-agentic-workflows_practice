"""One model call = one batch. Output is validated item by item; bad items are simply 'missing'."""
from __future__ import annotations

import json
from importlib import resources

from .schemas import SENTIMENTS, TOPICS, Extraction, Record

TOOL = {
    "name": "extract_batch", "description": "Return one extraction per input record.",
    "input_schema": {"type": "object", "required": ["results"], "properties": {"results": {
        "type": "array", "items": {"type": "object",
            "required": ["id", "sentiment", "topics", "summary", "confidence"],
            "properties": {
                "id": {"type": "string"},
                "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
                "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
                "summary": {"type": "string", "description": "At most 15 words."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}}}},
}


def system_prompt() -> str:
    return resources.files("data_pipeline.prompts").joinpath("extract.md").read_text(encoding="utf-8")


async def extract(client, records: list[Record], model: str, temperature: float = 0.0) -> dict[str, Extraction]:
    payload = json.dumps([{"id": r.id, "text": r.text} for r in records], ensure_ascii=False)
    resp = await client.complete(
        system=system_prompt(), messages=[{"role": "user", "content": f"<records>\n{payload}\n</records>"}],
        model=model, max_tokens=min(4000, 150 * len(records) + 200), temperature=temperature,
        tools=[TOOL], force_tool="extract_batch")
    by_id = {r.id: r for r in records}
    out: dict[str, Extraction] = {}
    for call in resp.tool_calls:
        if call.name != "extract_batch":
            continue
        for item in call.input.get("results", []):
            try:
                rec = by_id[str(item["id"])]
                sent = item["sentiment"]
                conf = float(item["confidence"])
                if sent not in SENTIMENTS or not 0 <= conf <= 1 or rec.id in out:
                    continue
                topics = [t for t in item.get("topics", []) if t in TOPICS] or ["other"]
                out[rec.id] = Extraction(rec.id, rec.seq, sent, sorted(set(topics)),
                                         str(item.get("summary", ""))[:200], conf, pii=rec.pii)
            except (KeyError, TypeError, ValueError):
                continue   # malformed or hallucinated id: treated as missing, recovered upstream
    return out
