"""Append-only checkpointing so a crashed or interrupted run resumes where it stopped."""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import Extraction, Failure


class Store:
    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.partial = self.dir / "results.partial.jsonl"
        self.dlq = self.dir / "failures.jsonl"
        self.final = self.dir / "results.jsonl"

    def done(self) -> dict[str, Extraction]:
        if not self.partial.exists():
            return {}
        out = {}
        for line in self.partial.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                out[d["id"]] = Extraction(**d)
            except (ValueError, KeyError, TypeError):
                continue   # a torn last line from a crash is ignored and redone
        return out

    def reset(self) -> None:
        for p in (self.partial, self.dlq, self.final):
            p.unlink(missing_ok=True)

    def append(self, items: list[Extraction]) -> None:
        with self.partial.open("a", encoding="utf-8") as f:
            for e in items:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            f.flush()

    def write_failures(self, fails: list[Failure]) -> None:
        with self.dlq.open("w", encoding="utf-8") as f:
            for x in fails:
                f.write(json.dumps(x.__dict__) + "\n")

    def finalize(self, items: list[Extraction]) -> None:
        with self.final.open("w", encoding="utf-8") as f:
            for e in sorted(items, key=lambda e: e.seq):
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
