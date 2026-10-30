"""Classification step using forced tool-use for schema-guaranteed output."""
from __future__ import annotations

from importlib import resources

from agentic_core import LLMClient

from .schemas import Classification, Route, Urgency

TOOL_NAME = "classify_ticket"
CLASSIFY_TOOL = {
    "name": TOOL_NAME,
    "description": "Record the classification of a customer support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": [r.value for r in Route]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "urgency": {"type": "string", "enum": [u.value for u in Urgency]},
            "reasoning": {"type": "string", "description": "One sentence."},
        },
        "required": ["route", "confidence", "urgency", "reasoning"],
    },
}


def load_prompt(name: str) -> str:
    return resources.files("support_router.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


async def classify(client: LLMClient, text: str, model: str) -> Classification:
    """Returns a validated Classification; on any failure falls back to a low-confidence GENERAL
    so the governance layer escalates to a human rather than guessing."""
    resp = await client.complete(
        system=load_prompt("classifier"),
        messages=[{"role": "user", "content": f"<ticket>\n{text}\n</ticket>"}],
        model=model, max_tokens=300, tools=[CLASSIFY_TOOL], force_tool=TOOL_NAME,
    )
    for call in resp.tool_calls:
        if call.name == TOOL_NAME:
            try:
                return Classification(**call.input)
            except (ValueError, TypeError):
                break
    return Classification(route=Route.GENERAL, confidence=0.0, reasoning="classifier_output_invalid")
