"""Deterministic report checks: every citation must resolve, and uncited prose is counted."""
from __future__ import annotations

import re

CITE = re.compile(r"\[(S\d+)\]")


def split_sentences(text: str) -> list[str]:
    # do not split before a citation marker, so "Claim. [S1]" stays one sentence
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?!\[S\d+\])", text) if s.strip()]


def check_text(text: str, valid: set[str]) -> tuple[str, int, int]:
    """Returns (text with unknown citations replaced by [?], invalid_count, uncited_sentence_count)."""
    invalid = 0

    def fix(m: re.Match) -> str:
        nonlocal invalid
        if m[1] in valid:
            return m[0]
        invalid += 1
        return "[?]"

    cleaned = CITE.sub(fix, text)
    uncited = sum(1 for s in split_sentences(cleaned) if not CITE.search(s))
    return cleaned, invalid, uncited
