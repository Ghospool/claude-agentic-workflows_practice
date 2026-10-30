"""Typed, validated domain objects (stdlib dataclasses: zero extra runtime dependencies)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SALES = "sales"
    GENERAL = "general"
    ESCALATION = "escalation"  # legal, safety, harassment, security incidents, press/exec


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip() or len(self.text) > 8000:
            raise ValueError("ticket text must be 1-8000 characters")


@dataclass(frozen=True)
class Classification:
    route: Route
    confidence: float
    urgency: Urgency = Urgency.MEDIUM
    reasoning: str = ""

    def __post_init__(self) -> None:
        # Coerce + validate untrusted model output; raises ValueError on anything off-schema.
        object.__setattr__(self, "route", Route(self.route))
        object.__setattr__(self, "urgency", Urgency(self.urgency))
        conf = float(self.confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "confidence", conf)


@dataclass
class RouterResult:
    ticket_id: str
    route: Route
    classification: Classification
    reply: str | None = None           # None => no automated reply was produced
    needs_human: bool = False
    human_reason: str | None = None
    pii_redacted: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=lambda o: o.value))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
