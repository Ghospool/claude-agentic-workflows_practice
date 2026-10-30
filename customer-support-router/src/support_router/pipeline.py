"""Orchestration: redact -> screen -> classify -> govern -> specialised handler -> audit."""
from __future__ import annotations

import asyncio
import logging

from agentic_core import LLMClient, Settings, get_settings
from agentic_core.security import AuditLog, looks_like_injection, redact_pii

from .classifier import classify, load_prompt
from .schemas import Route, RouterResult, Ticket

log = logging.getLogger(__name__)

HANDLER_PROMPTS = {
    Route.BILLING: "billing", Route.TECHNICAL: "technical", Route.ACCOUNT: "account",
    Route.SALES: "sales", Route.GENERAL: "general",
}


class SupportRouter:
    def __init__(self, client: LLMClient, settings: Settings | None = None,
                 confidence_threshold: float = 0.7, audit: AuditLog | None = None) -> None:
        self.client = client
        self.s = settings or get_settings()
        self.threshold = confidence_threshold
        self.audit = audit or AuditLog(self.s.audit_log_path)

    async def handle(self, ticket: Ticket) -> RouterResult:
        red = redact_pii(ticket.text)               # PII never reaches the model or the logs
        injected = looks_like_injection(red.text)
        before = (self.client.total_usage.input_tokens, self.client.total_usage.output_tokens)

        cls = await classify(self.client, red.text, self.s.fast_model)

        human_reason = None
        if injected:
            human_reason = "possible_prompt_injection"
        elif cls.route is Route.ESCALATION:
            human_reason = "escalation_category"
        elif cls.confidence < self.threshold:
            human_reason = f"low_confidence<{self.threshold}"

        reply = None
        if human_reason is None:
            handler = load_prompt(HANDLER_PROMPTS[cls.route])
            r = await self.client.complete(
                system=handler,
                messages=[{"role": "user", "content": f"<ticket>\n{red.text}\n</ticket>"}],
                model=self.s.smart_model, max_tokens=600, temperature=0.3,
            )
            reply = r.text

        # NOTE: token deltas are exact only when tickets are processed sequentially.
        result = RouterResult(
            ticket_id=ticket.id, route=cls.route, classification=cls, reply=reply,
            needs_human=human_reason is not None, human_reason=human_reason,
            pii_redacted=red.counts,
            input_tokens=self.client.total_usage.input_tokens - before[0],
            output_tokens=self.client.total_usage.output_tokens - before[1],
        )
        self.audit.write("ticket_routed", ticket_id=ticket.id, route=cls.route.value,
                         confidence=cls.confidence, needs_human=result.needs_human,
                         human_reason=human_reason, pii=red.counts, injection=injected)
        return result

    async def handle_many(self, tickets: list[Ticket], concurrency: int = 5) -> list[RouterResult]:
        sem = asyncio.Semaphore(concurrency)

        async def one(t: Ticket) -> RouterResult:
            async with sem:
                return await self.handle(t)

        return await asyncio.gather(*(one(t) for t in tickets))
