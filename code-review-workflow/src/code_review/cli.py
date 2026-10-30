"""code-review review <file.py> [--json] | code-review eval data [--offline]"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from agentic_core import AnthropicClient, get_settings

from .evaluate import run_eval
from .offline import offline_client
from .pipeline import CodeReviewer


def main() -> None:
    p = argparse.ArgumentParser(prog="code-review")
    p.add_argument("--offline", action="store_true", help="use the rule-based mock instead of Claude")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review"); r.add_argument("path"); r.add_argument("--json", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("data_dir")
    a = p.parse_args()

    s = get_settings()
    logging.basicConfig(level=s.log_level)
    reviewer = CodeReviewer(offline_client() if a.offline else AnthropicClient(s), s)

    async def run() -> int:
        if a.cmd == "review":
            rep = await reviewer.review_file(a.path)
            print(rep.to_json() if a.json else rep.to_markdown())
            return 1 if rep.verdict == "block" else 0   # non-zero exit gates CI on blockers
        res = await run_eval(reviewer, a.data_dir)
        for f in res.pop("files"):
            print(f"{f['file']:<22} expected={f['expected']} found={f['found']} verdict={f['verdict']}")
        print()
        for k, v in res.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        return 0

    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
