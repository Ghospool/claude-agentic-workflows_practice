# Research Assistant Agent (Autonomous agent pattern)

Answers a research question from your sources with a report in which every claim traces to a verified quote.

```
question -> PLAN (sub-questions) -> RESEARCH LOOP (model picks tools, parallel calls, hard budgets)
            search -> read_document -> record_finding (quote verified verbatim) -> finish_research
         -> SYNTHESISE from the evidence ledger only -> VERIFY citations -> report.md + reviewer notes
```

**What makes it an agent (not a workflow):** the model decides which tools to call, in what order, how many in parallel, and when it has enough. Code decides what it is *allowed* to do.

**Design decisions**
- **Evidence ledger:** only findings recorded through `record_finding` can reach the report. The executor rejects any quote that is not verbatim in a document the agent actually read; the rejection goes back to the agent as a tool error so it self-corrects.
- **Hard budgets:** max steps, max tool calls, cumulative token budget, and an idle-turn limit. If a budget stops the run, the report is still produced from verified evidence and says it stopped early.
- **No evidence, no report:** an empty ledger yields an explicit "insufficient evidence" note, never a fluent guess.
- **Citation verification:** unknown `[S#]` markers are replaced with `[?]` and counted; uncited sentences are counted and surfaced in reviewer notes. Conflicting sources must be listed under "Conflicting evidence".
- **Untrusted content:** documents are wrapped as data, scanned for instruction-like text (flagged in tool output and the report), and cannot expand the agent's permissions. The tool set is read-only.
- **Pluggable sources:** `LocalCorpusProvider` (folder of `.md`/`.txt`, BM25-style ranking), `CompositeProvider` (namespaced multi-source), `WebProvider` (bring your own search function, with an SSRF guard that blocks private/loopback/link-local addresses, re-validated on redirects, size and domain limits). The web fetch path is network code that I could not exercise offline; test it against your search API before relying on it.
- **Audit trail:** every tool call and run summary is written to a JSONL log.

**Run**
```bash
pip install -e ../agentic-core -e .                    # Python 3.10+
python -m unittest discover -s tests                   # 18 offline tests
research-agent --offline eval data
research-agent --offline run "What range limitations and battery resale risks should we plan for with electric vans?" --corpus data/corpus
cp .env.example .env                                   # add ANTHROPIC_API_KEY, then drop --offline
research-agent run "your question" --corpus path/to/your/docs --out report.md
```

**Evaluation** (`data/questions.jsonl`, synthetic corpus including a prompt-injection document, a conflicting analysis and an irrelevant distractor): expected-fact recall, invalid citations, uncited sentences, injection leaks, early stops, steps, tool calls, rejected findings and tokens. The offline policy is a scripted heuristic that exercises the loop; its output quality (for example awkward section headings and loosely relevant findings) says nothing about Claude's. The corpus is fictional. Measure real quality with a live key and your own documents and questions.

**Known limits:** fact recall is checked by string match, not by an LLM judge or human review; no claim-level entailment check beyond the verbatim-quote rule; one question per run, no memory across runs; context is not summarised, so very long runs rely on the step and token budgets.
