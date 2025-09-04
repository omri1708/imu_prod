# synth/rollout.py
from __future__ import annotations
from typing import Dict, Any

def gate(canary_result: Dict[str,Any]) -> Dict[str,Any]:
    if canary_result.get("ok"):
        return {"approved": True, "policy":"safe-progress"}
    return {"approved": False, "policy":"rollback"}