# imu_repo/engine/ab_selector.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time

from grounded.claims import current
from engine.phi_multi import phi_score
from user_model.policy import get_profile

def _simulate_perf(variant: Dict[str,Any]) -> Dict[str,float]:
    """
    הערכת ביצועים דטרמיניסטית:
    - תג #FAST מוריד לטנטיות/עלות/אנרגיה.
    - תג #SLOW מעלה אותם.
    - אורך הקוד משפיע על cost_units/mem_kb.
    """
    code: str = str(variant.get("code",""))
    fast = "#FAST" in code
    slow = "#SLOW" in code

    base_p95 = 40.0 if fast else (400.0 if slow else 120.0)   # ms
    # לייצב: תלות באורך קוד
    klen = max(1, len(code))
    cost_units = float(klen)
    mem_kb = float(min(5120, 0.02 * klen))   # KB ~ ביחס לאורך
    energy_units = float(min(200.0, 0.0008 * klen * (2.0 if slow else 1.0)))

    error_rate = 0.01 if fast else (0.03 if slow else 0.02)   # דמה דטרמיניסטי
    source_trust = 0.92 if fast else (0.88 if slow else 0.90) # מדמה איכות ראיות

    return {
        "p95_ms": base_p95,
        "cost_units": cost_units,
        "error_rate": error_rate,
        "source_trust": source_trust,
        "energy_units": energy_units,
        "mem_kb": mem_kb,
    }

def _score_variant(v: Dict[str,Any], weights: Dict[str,float]) -> Tuple[float, Dict[str,float]]:
    perf = _simulate_perf(v)
    phi = phi_score(perf, weights)
    return phi, perf

def select_best(variants: List[Dict[str,Any]], *, spec: Dict[str,Any] | None = None, user_id: str = "default") -> Dict[str,Any]:
    """
    בוחר וריאציה מנצחת לפי Φ מרובה־יעדים ומשקולות פרופיל משתמש.
    מחזיר:
      {
        "winner": {"label": "...", "language": "...", "code": "..."},
        "info": {
            "phi": float,
            "perf": {...},
            "alternatives": [{"label": ..., "phi": ...}, ...]
        }
      }
    """
    prof = get_profile(user_id)
    weights = dict(prof.get("phi_weights", {}))  # אם לא קיים → ריק → DEFAULT_WEIGHTS יחולו פנימה

    scored: List[Tuple[float, Dict[str,float], Dict[str,Any]]] = []
    for v in variants:
        phi, perf = _score_variant(v, weights)
        scored.append((phi, perf, v))

    scored.sort(key=lambda x: x[0])  # קטן יותר טוב
    best_phi, best_perf, best_v = scored[0]

    # Evidences על האלטרנטיבות וההחלטה
    current().add_evidence("ab_decision", {
        "source_url": "local://ab",
        "trust": 0.95,
        "ttl_s": 900,
        "payload": {
            "weights": weights,
            "chosen": {"label": best_v.get("label"), "phi": best_phi, "perf": best_perf},
            "alternatives": [
                {"label": v.get("label"), "phi": float(ph), "p95_ms": float(p["p95_ms"])}
                for ph, p, v in scored[1:]
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