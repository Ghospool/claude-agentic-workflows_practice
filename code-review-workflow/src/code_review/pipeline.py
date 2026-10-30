"""Chain: redact -> analyze -> [ground gate] -> verify -> [confidence gate] -> suggest -> summarize.

Deterministic gates between model calls are the point of the pattern: each step's output is
validated before it becomes the next step's input, so errors do not compound."""
from __future__ import annotations

import logging
from pathlib import Path

from agentic_core import LLMClient, Settings, get_settings
from agentic_core.security import AuditLog, looks_like_injection, redact_secrets

from . import steps
from .schemas import Issue, ReviewReport, Severity, StepStats

log = logging.getLogger(__name__)
MAX_LINES = 600


class CodeReviewer:
    def __init__(self, client: LLMClient, settings: Settings | None = None, min_confidence: float = 0.6,
                 fix_min_severity: Severity = Severity.LOW, audit: AuditLog | None = None) -> None:
        self.client = client
        self.s = settings or get_settings()
        self.min_confidence = min_confidence
        self.fix_min_severity = fix_min_severity
        self.audit = audit or AuditLog(self.s.audit_log_path)

    async def review_file(self, path: str | Path) -> ReviewReport:
        p = Path(path)
        return await self.review(p.name, p.read_text(encoding="utf-8"))

    async def review(self, filename: str, source: str) -> ReviewReport:
        warnings: list[str] = []
        red = redact_secrets(source)
        src = red.text
        n_lines = len(src.splitlines())
        if n_lines > MAX_LINES:
            warnings.append(f"file truncated to first {MAX_LINES} of {n_lines} lines")
            src = "\n".join(src.splitlines()[:MAX_LINES])
        if looks_like_injection(src):
            warnings.append("source contains instruction-like text aimed at AI reviewers; treated as data")
        if red.counts.get("SECRET"):
            warnings.append(f"{red.counts['SECRET']} hard-coded secret(s) detected and redacted; rotate them")

        u0 = (self.client.total_usage.input_tokens, self.client.total_usage.output_tokens)
        stats = StepStats()
        lines = src.splitlines()

        # Step 1: analyze
        cands = await steps.analyze(self.client, filename, src, self.s.smart_model)
        stats.proposed = len(cands)

        # Gate 1: grounding. A finding must point at a real line whose text matches the evidence.
        grounded: list[Issue] = []
        for c in cands:
            ok = 1 <= c.line <= len(lines) and c.evidence.strip() and c.evidence.strip() in lines[c.line - 1]
            if ok:
                grounded.append(c)
            else:
                stats.dropped_ungrounded += 1
        for idx, g in enumerate(grounded, 1):
            g.id = f"I{idx}"

        # Step 2: independent verification pass (skeptical reviewer) to cut false positives
        confirmed = grounded
        if grounded:
            try:
                verdicts = await steps.verify(self.client, filename, src, grounded, self.s.smart_model)
            except ValueError:
                verdicts, _ = {}, warnings.append("verification step failed; findings are unverified")
            confirmed = []
            for g in grounded:
                v = verdicts.get(g.id)
                if v is not None and not v.get("valid", False):
                    stats.dropped_rejected += 1
                    continue
                if v and v.get("severity"):
                    try:
                        g.severity = Severity(v["severity"])
                    except ValueError:
                        pass
                # Gate 2: confidence floor
                if g.confidence < self.min_confidence:
                    stats.dropped_low_confidence += 1
                    continue
                confirmed.append(g)
        stats.kept = len(confirmed)

        # Step 3: fixes (skipped entirely when nothing survived: no wasted tokens)
        fixable = [i for i in confirmed if i.severity.rank >= self.fix_min_severity.rank]
        if fixable:
            try:
                fixes = await steps.suggest(self.client, filename, src, fixable, self.s.smart_model)
                for i in fixable:
                    f = fixes.get(i.id)
                    if f:
                        i.fix_description, i.patch = f.get("fix_description", ""), f.get("patch", "")
            except ValueError:
                warnings.append("fix suggestion step failed; issues reported without patches")

        # Step 4: summary + deterministic verdict policy (never delegated to the model)
        summary = (await steps.summarize(self.client, filename, confirmed, self.s.fast_model)
                   if confirmed else "No actionable issues were found.")
        top = max((i.severity.rank for i in confirmed), default=-1)
        verdict = "block" if top >= Severity.CRITICAL.rank else "request_changes" if top >= Severity.HIGH.rank or \
            (confirmed and top >= Severity.MEDIUM.rank) else "approve"

        report = ReviewReport(
            filename=filename, verdict=verdict, summary=summary, issues=confirmed, stats=stats,
            warnings=warnings, secrets_redacted=red.counts.get("SECRET", 0),
            input_tokens=self.client.total_usage.input_tokens - u0[0],
            output_tokens=self.client.total_usage.output_tokens - u0[1])
        self.audit.write("code_reviewed", file=filename, verdict=verdict, kept=stats.kept,
                         proposed=stats.proposed, secrets=report.secrets_redacted)
        return report
