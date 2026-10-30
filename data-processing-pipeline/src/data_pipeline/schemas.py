from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SENTIMENTS = ("positive", "neutral", "negative")
TOPICS = ("pricing", "quality", "delivery", "support", "usability", "reliability", "other")


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    seq: int = 0                       # position in the input; restores order after parallel processing
    pii: dict = field(default_factory=dict)


@dataclass
class Extraction:
    id: str
    seq: int
    sentiment: str
    topics: list[str]
    summary: str
    confidence: float
    votes: dict[str, int] = field(default_factory=dict)   # sentiment -> count, only when voting ran
    needs_review: bool = False
    pii: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Failure:
    id: str
    seq: int
    reason: str


@dataclass
class RunSummary:
    input_records: int = 0
    rejected: int = 0
    duplicates: int = 0
    skipped_resumed: int = 0
    processed: int = 0
    failed: int = 0
    needs_review: int = 0
    voted: int = 0
    api_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    estimated_cost_usd: float | None = None
    rejection_samples: list[str] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
