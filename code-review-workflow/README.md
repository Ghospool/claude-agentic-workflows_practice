# Code Review Workflow (Prompt-chaining pattern)

```
source -> redact secrets -> ANALYZE (Sonnet, forced tool call)
       -> GATE 1: grounding (line exists, evidence matches the code)
       -> VERIFY (independent skeptical pass; may reject or re-rate)
       -> GATE 2: confidence floor
       -> SUGGEST FIXES (only for surviving issues)
       -> SUMMARISE (Haiku) -> deterministic verdict policy -> report + audit log
```

**Design decisions**
- Gates between model calls are code, not prompts: hallucinated findings (wrong line, evidence not in the file) never reach later steps.
- A second, independent verification call trades latency and tokens for fewer false positives, the core chaining trade-off.
- Clean code short-circuits after step 1, so no tokens are spent on verify/fix/summary.
- The verdict (`approve` / `request_changes` / `block`) is computed from severities by policy, never by the model. `block` returns exit code 1, so the CLI can gate CI.
- Secrets are masked before any model call and before reporting; instruction-like text in comments is flagged and treated as data.
- Any step failure degrades gracefully with a visible warning instead of crashing or silently passing.

**Run**
```bash
pip install -e ../agentic-core -e .                      # Python 3.10+
python -m unittest discover -s tests                     # offline tests
code-review --offline eval data                          # eval on seeded-bug samples
code-review --offline review data/samples/auth_service.py
cp .env.example .env                                     # add ANTHROPIC_API_KEY, then drop --offline
code-review review path/to/file.py [--json]
```

**Evaluation** (`data/samples` + `data/expected.jsonl`): line-level precision, recall and F1 (±1 line tolerance), false positives on a clean file, and secret leakage into reports. Labels are kept out of the sample files so the model cannot see them. The offline engine is rule-based, so its scores only validate the plumbing; run the eval with a live key and extend the samples with real, anonymised pull-request diffs before relying on it.

**Known limits:** reviews whole files (first 600 lines), not diffs; Python-oriented prompts; no cross-file context.
