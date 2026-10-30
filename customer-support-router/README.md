# Customer Support Router (Routing pattern)

```
ticket -> redact PII -> injection screen -> classify (Haiku, forced tool call)
       -> governance gate -> specialised handler prompt (Sonnet) -> audit log
                 |
                 +-> human queue if: escalation category | injection | confidence < 0.7 | invalid output
```

**Design decisions**
- Structured output via forced tool use, validated in code; invalid output fails safe to a human, never a guess.
- Cheap/fast model classifies; stronger model only writes replies for tickets that pass the gate.
- Escalation topics (legal, security, press, regulatory) are never auto-answered.
- PII is redacted before any model call or log write; ticket text is treated as data, not instructions.
- Prompts live in `prompts/*.md` so they can be reviewed and versioned without code changes.

**Run**
```bash
pip install -e ../agentic-core -e .    # Python 3.10+
python -m unittest discover -s tests   # offline tests
support-router --offline eval data/tickets_golden.jsonl
cp .env.example .env                   # add ANTHROPIC_API_KEY, then drop --offline
support-router route "I was charged twice this month"
```

**Evaluation** (`data/tickets_golden.jsonl`, 24 labelled tickets): accuracy, per-class P/R/F1, escalation recall (target 100%), automation precision, human-handoff rate, PII coverage, token usage. The offline mock is keyword-based, so its scores only prove the plumbing; measure real quality with a live key, and grow the dataset with real (anonymised) tickets before any production use.
