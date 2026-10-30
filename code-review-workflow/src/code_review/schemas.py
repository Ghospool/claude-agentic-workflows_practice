from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return ["low", "medium", "high", "critical"].index(self.value)


class Category(str, Enum):
    SECURITY = "security"
    BUG = "bug"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


@dataclass
class Issue:
    line: int
    category: Category
    severity: Severity
    title: str
    explanation: str
    evidence: str            # verbatim code from the flagged line; used to ground the finding
    confidence: float = 0.8
    id: str = ""
    fix_description: str = ""
    patch: str = ""

    def __post_init__(self) -> None:
        self.category = Category(self.category)
        self.severity = Severity(self.severity)
        self.line = int(self.line)
        self.confidence = float(self.confidence)
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence out of range")


@dataclass
class StepStats:
    proposed: int = 0
    dropped_ungrounded: int = 0
    dropped_rejected: int = 0
    dropped_low_confidence: int = 0
    kept: int = 0


@dataclass
class ReviewReport:
    filename: str
    verdict: str                      # "approve" | "request_changes" | "block"
    summary: str
    issues: list[Issue] = field(default_factory=list)
    stats: StepStats = field(default_factory=StepStats)
    warnings: list[str] = field(default_factory=list)
    secrets_redacted: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=lambda o: o.value))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        icon = {"approve": "APPROVE", "request_changes": "REQUEST CHANGES", "block": "BLOCK"}[self.verdict]
        out = [f"# Code review: `{self.filename}`", "", f"**Verdict: {icon}**", "", self.summary, ""]
        for w in self.warnings:
            out += [f"> Warning: {w}", ""]
        if not self.issues:
            out.append("No issues found.")
        for i in sorted(self.issues, key=lambda x: (-x.severity.rank, x.line)):
            out += [f"## [{i.severity.value.upper()}] {i.title} (line {i.line}, {i.category.value})", "",
                    i.explanation, "", f"Evidence: `{i.evidence}`", ""]
            if i.fix_description:
                out += [f"**Suggested fix:** {i.fix_description}", ""]
            if i.patch:
                out += ["```python", i.patch, "```", ""]
        s = self.stats
        out += ["---", f"Pipeline: {s.proposed} proposed, {s.dropped_ungrounded} dropped (ungrounded), "
                f"{s.dropped_rejected} rejected on verification, {s.dropped_low_confidence} low confidence, "
                f"{s.kept} kept. Secrets redacted: {self.secrets_redacted}. "
                f"Tokens: {self.input_tokens} in / {self.output_tokens} out."]
        return "\n".join(out)
