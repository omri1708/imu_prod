# imu_repo/engine/ab_selector.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time

from grounded.claims import current
from engine.phi_multi import phi_score
from engine.phi_multi_context import effective_weights
from engine.pareto import pareto_front
from user_model.policy import get_profile

def _simulate_perf(variant: Dict[str,Any]) -> Dict[str,float]:
    code: str = str(variant.get("code",""))
    fast = "#FAST" in code
    slow = "#SLOW" in code
    explore = "#EXPLORE" in code

    base_p95 = 40.0 if fast else (400.0 if slow else 120.0)   # ms
    klen = max(1, len(code))
    cost_units = float(klen)
    mem_kb = float(min(5120, 0.02 * klen))
    energy_units = float(min(200.0, 0.0008 * klen * (2.0 if slow else 1.0)))

    # ענישת יציבות קלה ל-EXPLORE: שגיאות מעט גבוהות יותר ואמון מעט נמוך יותר
    base_err = 0.01 if fast else (0.03 if slow else 0.02)
    base_trust = 0.92 if fast else (0.88 if slow else 0.90)
    if explore:
        base_err += 0.005
        base_trust -= 0.02
        if base_trust < 0.5: base_trust = 0.5

    return {
        "p95_ms": base_p95,
        "cost_units": cost_units,
        "error_rate": base_err,
        "source_trust": base_trust,
        "energy_units": energy_units,
        "mem_kb": mem_kb,
    }

def _metrics_vector(perf: Dict[str,float]) -> List[float]:
    # וקטור למינימיזציה עבור Pareto:
    return [
        float(perf["p95_ms"]),
        float(perf["cost_units"]),
        float(perf["error_rate"]),
        float(1.0 - perf["source_trust"]),
        float(perf["energy_units"]),
        float(perf["mem_kb"]),
    ]

def select_best(variants: List[Dict[str,Any]], *, spec: Dict[str,Any] | None = None,
                user_id: str = "default", intents: List[str] | None = None) -> Dict[str,Any]:
    prof = get_profile(user_id)
    user_weights = dict(prof.get("phi_weights", {}))
    eff_weights = effective_weights(user_weights, intents or [])

    # חישוב מטריקות+Φ לכל וריאציה
    scored: List[Tuple[float, Dict[str,float], Dict[str,Any]]] = []
    vectors: List[List[float]] = []
    for v in variants:
        perf = _simulate_perf(v)
        phi = phi_score(perf, eff_weights)
        vectors.append(_metrics_vector(perf))
        scored.append((phi, perf, v))

    # חזית Pareto
    frontier_idx = set(pareto_front(vectors))
    frontier = [scored[i] for i in frontier_idx]
    # בוחרים מנצח על החזית לפי Φ (המשוקלל-הקשר)
    frontier.sort(key=lambda x: x[0])
    best_phi, best_perf, best_v = frontier[0]

    # Evidences
    current().add_evidence("ab_decision_ctx_pareto", {
        "source_url": "local://ab_ctx_pareto",
        "trust": 0.95,
        "ttl_s": 900,
        "payload": {
            "intents": intents or [],
            "weights_effective": eff_weights,
            "frontier_size": len(frontier),
            "chosen": {"label": best_v.get("label"), "phi": float(best_phi),
                       "p95_ms": float(best_perf["p95_ms"]),
                       "cost_units": float(best_perf["cost_units"]),
                       "error_rate": float(best_perf["error_rate"]),
                       "source_trust": float(best_perf["source_trust"])},
            "alternatives_on_frontier": [
                {"label": v.get("label"), "phi": float(ph)}
                for ph, p, v in frontier[1:]
            ]
        }
    })

    return {
        "winner": {
            "label": best_v.get("label"),
            "language": best_v.get("language"),
            "code": best_v.get("code"),
        },
        "info": {
            "phi": float(best_phi),
            "perf": {
                "p95_ms": float(best_perf["p95_ms"]),
                "error_rate": float(best_perf["error_rate"]),
                "cost_units": float(best_perf["cost_units"]),
                "energy_units": float(best_perf["energy_units"]),
                "mem_kb": float(best_perf["mem_kb"]),
            }
        }
    }