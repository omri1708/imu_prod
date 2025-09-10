from __future__ import annotations
from typing import Any, Dict

def make_proof(*, citations, model, p95_ms=None, gates=None, cost_usd=None, agg_trust=None) -> Dict[str,Any]:
    return {
        "citations": citations or [],
        "model": model,
        "p95_ms": p95_ms,
        "gates": gates or {},
        "cost_usd": cost_usd,
        "agg_trust": agg_trust
    }
