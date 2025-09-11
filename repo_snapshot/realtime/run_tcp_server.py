# imu_repo/realtime/run_tcp_server.py
from __future__ import annotations
import asyncio
from typing import Dict, Any, Tuple
from realtime.server import run_tcp

DEFAULT_POLICY = {
    "min_distinct_sources": 1,
    "min_total_trust": 1.0,
    "perf_sla": {"latency_ms": {"p95_max": 200.0}},
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9401)
    args = ap.parse_args()
    asyncio.run(run_tcp(args.host, args.port, DEFAULT_POLICY))