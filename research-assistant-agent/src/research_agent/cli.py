"""research-agent run "question" --corpus data/corpus [--out report.md] | research-agent eval data [--offline]"""
from __future__ import annotations

import argparse
import asyncio
import logging

from agentic_core import AnthropicClient, get_settings

from .agent import AgentConfig, ResearchAgent
from .evaluate import run_eval
from .offline import offline_client
from .sources import LocalCorpusProvider


def main() -> None:
    p = argparse.ArgumentParser(prog="research-agent")
    p.add_argument("--offline", action="store_true", help="use the scripted mock instead of Claude")
    p.add_argument("--max-steps", type=int, default=10)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("question"); r.add_argument("--corpus", required=True); r.add_argument("--out")
    e = sub.add_parser("eval"); e.add_argument("data_dir")
    a = p.parse_args()
    s = get_settings(); logging.basicConfig(level=s.log_level)
    client = offline_client() if a.offline else AnthropicClient(s)

    async def go() -> None:
        if a.cmd == "run":
            agent = ResearchAgent(client, LocalCorpusProvider(a.corpus), AgentConfig(max_steps=a.max_steps), s)
            res = await agent.run(a.question)
            if a.out:
                open(a.out, "w", encoding="utf-8").write(res.report_markdown)
            print(res.report_markdown); print("\n", res.metrics())
        else:
            for k, v in (await run_eval(client, a.data_dir, s)).items():
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    asyncio.run(go())


if __name__ == "__main__":
    main()
