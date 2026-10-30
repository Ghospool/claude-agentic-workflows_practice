from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Finding:
    id: str
    sub_question: int
    claim: str
    source_id: str
    quote: str            # verbatim text from the source; verified at record time


@dataclass
class Source:
    id: str               # S1, S2... assigned when a document is first read
    doc_id: str
    title: str
    url: str = ""
    text: str = ""        # full text kept for quote verification


@dataclass
class ResearchResult:
    question: str
    sub_questions: list[str]
    report_markdown: str
    findings: list[Finding] = field(default_factory=list)
    sources: dict[str, dict[str, str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stop_reason: str = "finished"
    steps: int = 0
    tool_calls: int = 0
    rejected_findings: int = 0
    invalid_citations: int = 0
    uncited_sentences: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def metrics(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in ("stop_reason", "steps", "tool_calls", "rejected_findings",
                                  "invalid_citations", "uncited_sentences", "input_tokens", "output_tokens")}
