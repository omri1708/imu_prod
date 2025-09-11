# imu_repo/engine/phi_multi_context.py
from __future__ import annotations
from typing import Dict, Any, List
from engine.phi_multi import DEFAULT_WEIGHTS

# התאמות (דלתא) לפי Intent; ערכים שאינם קיימים ב-DEFAULT לא נלקחים.
INTENT_DELTAS: Dict[str, Dict[str, float]] = {
    "realtime":  {"latency": +0.25, "errors": +0.05, "distrust": +0.02, "cost": -0.15},
    "batch":     {"latency": -0.20, "cost": +0.15, "energy": +0.05},
    "mobile":    {"energy": +0.10, "latency": +0.05, "memory": +0.05},
    "sensitive": {"errors": +0.20, "distrust": +0.10, "cost": -0.10},
    "cost_saver":{"cost": +0.30, "latency": -0.20},
    "gpu":       {"energy": +0.06, "cost": +0.08, "latency": -0.06},
    "ui":        {"latency": +0.08, "errors": +0.04},
}

def _merge_weights(base: Dict[str,float], extra: Dict[str,float]) -> Dict[str,float]:
    out = dict(base)
    for k,v in extra.items():
        if k in out:
            out[k] = float(out[k]) + float(v)
    # לא נורמליזציה קשיחה—Φ הוא סכום משוקלל; המשקולות היחסיות הן החשובות.
    # דואגים שלא יהיו שליליים:
    for k in list(out.keys()):
        if out[k] < 0.0:
            out[k] = 0.0
    return out

def effective_weights(user_weights: Dict[str,float] | None, intents: List[str]) -> Dict[str,float]:
    w = dict(DEFAULT_WEIGHTS)
    if user_weights:
        w = _merge_weights(w, {k: float(v) for k,v in user_weights.items() if k in w})
    for tag in intents:
        delta = INTENT_DELTAS.get(tag, {})
        w = _merge_weights(w, delta)
    return w