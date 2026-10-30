# Claude Agentic Workflows

Four production-style implementations of the core agentic design patterns, built on the Claude API with Python. Each project is self-contained, tested offline, and ships with an evaluation harness and a design write-up.

| Project | Pattern | What it does | Engineering highlights |
|---|---|---|---|
| [`customer-support-router`](customer-support-router) | Routing | Classifies support tickets and answers them with specialised prompts | Forced tool-call classification, confidence gate, human hand-off, PII redaction, injection screening, audit log |
| [`code-review-workflow`](code-review-workflow) | Prompt chaining | Analyses code, verifies findings, proposes fixes, reports | Deterministic gates between steps, grounding check against hallucinated findings, secret redaction, CI-friendly exit codes |
| [`data-processing-pipeline`](data-processing-pipeline) | Parallelization | Extracts structured data from large JSONL files with concurrent calls | Bounded concurrency, rate limiting, jittered retries, split-on-failure, dead-letter queue, checkpoint/resume, voting |
| [`research-assistant-agent`](research-assistant-agent) | Autonomous agent | Researches a question across sources and writes a cited report | Tool loop with hard budgets, verbatim-quote evidence ledger, citation verification, SSRF-guarded web access |
| [`agentic-core`](agentic-core) | Shared library | Typed async client, mock client, security helpers, audit log, eval metrics | Swap real and mock clients without touching workflow code |

A plain-language walkthrough is in [`docs/Project_Overview.pdf`](docs/Project_Overview.pdf).

## Design principles

- **Code decides what is allowed; the model decides what is useful.** Validation, budgets, verdict policy and escalation rules are ordinary code, never prompts.
- **Fail safe.** Invalid model output, failed steps and low confidence lead to human review or an explicit warning, never a silent guess.
- **Untrusted input stays data.** PII and secrets are redacted before model calls; documents, tickets and code are screened for injection and treated strictly as data.
- **Measure, do not assume.** Every project has a labelled evaluation set with task-specific metrics, including safety metrics such as escalation recall and secret leakage.
- **Testable without an API key.** All projects run offline against a mock client, so the full suite runs in CI.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/Ghospool/claude-agentic-workflows_practice.git && cd claude-agentic-workflows_practice
python run_tests.py                       # all offline suites, no API key needed
```

Run one project (example, macOS/Linux; on Windows PowerShell use `;` instead of `&&` and `.venv\Scripts\Activate.ps1`):

```bash
python -m venv .venv && source .venv/bin/activate
cd customer-support-router
pip install -e ../agentic-core -e .
support-router --offline eval data/tickets_golden.jsonl
```

To use real Claude models, copy `.env.example` to `.env` inside the project folder, set `ANTHROPIC_API_KEY`, and drop `--offline`. Model names are configurable in `.env`.

## Verification status

- All four test suites (54 tests) pass offline, including fault-injection tests for retries, partial failures, resume after a crash, budget limits and prompt-injection handling.
- The offline "model" in each project is a deterministic stand-in (keyword rules or a scripted policy). It validates the plumbing and safety logic, **not** the quality of Claude's answers. The scores printed by `--offline eval` therefore should not be read as model accuracy.

## Known limitations

Sample datasets are small and synthetic. The research agent's web provider needs your own search function and has not been exercised against live sites. Prompts and thresholds are starting points to tune on real data. See each project's README for specifics.

## Repository layout

```
agentic-core/               shared client, mock, security, audit, eval metrics
customer-support-router/    routing pattern
code-review-workflow/       chaining pattern
data-processing-pipeline/   parallelization pattern
research-assistant-agent/   autonomous agent
docs/                       project overview (PDF)
run_tests.py                runs every offline suite
.github/workflows/ci.yml    CI on Python 3.10-3.12
```

## License

MIT. See [LICENSE](LICENSE).
