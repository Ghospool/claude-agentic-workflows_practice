"""CLI: support-router route "text" | support-router eval data/tickets_golden.jsonl [--offline]"""
from __future__ import annotations

import argparse
import asyncio
import logging

from agentic_core import AnthropicClient, get_settings

from .evaluate import run_eval
from .offline import offline_client
from .pipeline import SupportRouter
from .schemas import Ticket


def main() -> None:
    p = argparse.ArgumentParser(prog="support-router")
    p.add_argument("--offline", action="store_true", help="use the keyword mock instead of Claude")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("route"); r.add_argument("text")
    e = sub.add_parser("eval"); e.add_argument("dataset")
    a = p.parse_args()

    s = get_settings()
    logging.basicConfig(level=s.log_level)
    client = offline_client() if a.offline else AnthropicClient(s)
    router = SupportRouter(client, s)

    async def run() -> None:
        if a.cmd == "route":
            print((await router.handle(Ticket(id="cli", text=a.text))).to_json())
        else:
            report, extra = await run_eval(router, a.dataset)
            print(report.to_markdown(), "\n")
            for k, v in extra.items():
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
