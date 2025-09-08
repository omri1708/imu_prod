#!/usr/bin/env python3
from __future__ import annotations
import json, sys, asyncio, argparse
from pathlib import Path
from engine.pipelines.orchestrator import Orchestrator, default_runners

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to JSON file, or '-' for stdin")
    ap.add_argument("--ctx", default=None, help="optional path to ctx JSON")
    args = ap.parse_args()

    if args.spec == "-":
        spec = json.load(sys.stdin)
    else:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    ctx = {}
    if args.ctx:
        ctx = json.loads(Path(args.ctx).read_text(encoding="utf-8"))
    ctx.setdefault("user_id", "anon")

    orch = Orchestrator(default_runners())
    out = await orch.run_any(spec, ctx)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
