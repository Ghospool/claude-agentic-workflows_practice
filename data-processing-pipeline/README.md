# Data Processing Pipeline (Parallelization pattern)

Turns large files of unstructured text (here: customer feedback) into validated, structured records using many concurrent Claude calls.

```
JSONL input -> stream, validate, dedupe, redact PII -> batches of 20
  -> bounded queue (backpressure) -> N workers -> Gateway (concurrency cap + rate limit)
  -> extract_batch (forced tool call, per-item validation)
        |-- transient error  -> retry with jittered exponential backoff
        |-- missing/invalid items -> re-run only those
        |-- hard failure -> split batch in half, recurse; a single bad record ends in the dead-letter queue
  -> low-confidence records: VOTE (2 extra runs, majority wins; ties go to a human)
  -> append-only checkpoint -> sorted results.jsonl + summary.json + report.md
```

**Patterns used:** sectioning (independent batches in parallel) and voting (self-consistency on uncertain records).

**Design decisions**
- Batching ~20 records per call cuts calls and cost roughly 10x versus one call per record, at the price of larger blast radius, which the split-on-failure logic contains.
- A single `Gateway` owns concurrency and request rate, so the limits hold even while voting fans out extra calls.
- A bounded queue streams the file; memory stays flat for very large inputs (only ids are held for dedupe).
- Every completed batch is checkpointed immediately. Re-running the same command resumes; a torn last line from a crash is ignored.
- Items are validated individually: unknown ids, bad enums or out-of-range confidence are treated as missing and recovered, never trusted.
- Nothing is silently lost: rejected lines, duplicates, failures (with reasons) and human-review flags are all counted in `summary.json`.
- PII is redacted before the model sees the text; the audit log records counts only.
- Cost: pass `price_in_per_mtok` / `price_out_per_mtok` in `PipelineConfig` from current pricing to get `estimated_cost_usd` (no prices are hard-coded because they change).

**Run**
```bash
pip install -e ../agentic-core -e .                 # Python 3.10+
python -m unittest discover -s tests                # 13 offline tests incl. fault injection
python data/generate_sample_data.py                 # regenerates 300 labelled reviews
data-pipeline --offline eval data
data-pipeline --offline run data/reviews.jsonl out/ # resumable; add --no-resume to restart
cp .env.example .env                                # add ANTHROPIC_API_KEY, then drop --offline
```

**Evaluation** (`data/reviews.jsonl` + `labels.jsonl`, includes sarcasm and negation cases): sentiment accuracy and macro F1, accuracy of auto-accepted records (the number that matters for trust), human-review rate, topic precision/recall/F1, failures, records per call and tokens. The offline lexicon engine is a plumbing check only; its misses on sarcasm show why the review queue exists. Measure real quality with a live key.

**Known limits:** JSONL input only; the taxonomy is fixed in code; voting is per record, not per field; for very large offline jobs consider the Message Batches API (asynchronous, discounted) in place of real-time calls.
