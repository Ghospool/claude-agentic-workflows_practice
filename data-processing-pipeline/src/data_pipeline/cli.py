"""data-pipeline run <in.jsonl> <out_dir> [--no-resume] | data-pipeline eval <data_dir> [--offline]"""
from __future__ import annotations

import argparse
import asyncio
import logging

from agentic_core import AnthropicClient, get_settings

from .evaluate import run_eval
from .offline import offline_client
from .pipeline import DataPipeline, PipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(prog="data-pipeline")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--rps", type=float, default=20.0, help="max API requests per second")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("input"); r.add_argument("out_dir"); r.add_argument("--no-resume", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("data_dir")
    a = p.parse_args()

    s = get_settings()
    logging.basicConfig(level=s.log_level)
    cfg = PipelineConfig(batch_size=a.batch_size, concurrency=a.concurrency, requests_per_second=a.rps)
    pipe = DataPipeline(offline_client() if a.offline else AnthropicClient(s), cfg, s)

    async def run() -> None:
        if a.cmd == "run":
            sm = await pipe.run(a.input, a.out_dir, resume=not a.no_resume)
            print(sm.to_json())
        else:
            for k, v in (await run_eval(pipe, a.data_dir)).items():
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
