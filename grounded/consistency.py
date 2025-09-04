# imu_repo/grounded/consistency.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

from grounded.provenance import ProvenanceStore
from grounded.contradiction_policy import policy_singleton as Policy, MetricRule

def _extract_metrics(key: str, payload: Dict[str,Any]) -> Dict[str,float]:
    """
    ממפה Payload → מדדי־על השמישים להשוואה צולבת.
    מפתחות נתמכים כיום:
      - perf_summary: {"p95_ms": ...}
      - ui_accessibility: {"score": ...}
      - service_tests: {"passed": True/False}
      - db_migration: {"out":[...]} → rows (אם יש SELECT COUNT(*) ... AS n)
      - כל evidence חיצונית: ננסה לחלץ פרמטרים נפוצים אם קיימים
    """
    out: Dict[str,float] = {}
    if key == "perf_summary":
        p95 = payload.get("p95_ms")
        if isinstance(p95, (int,float)): out["perf.p95_ms"] = float(p95)
    elif key == "ui_accessibility":
        sc = payload.get("score")
        if isinstance(sc, (int,float)): out["ui.score"] = float(sc)
    elif key == "service_tests":
        passed = payload.get("passed")
        if isinstance(passed, bool): out["tests.passed"] = 1.0 if passed else 0.0
    elif key == "db_migration":
        # payload = {"out":[...]} , מנסים למצוא dict עם {"n": <rows>}
        outrows = payload.get("out")
        if isinstance(outrows, list):
            n = None
            for item in outrows:
                if isinstance(item, dict) and "n" in item:
                    try:
                        n = float(item["n"])
                        break
                    except Exception:
                        pass
            if isinstance(n,(int,float)): out["db.rows"] = float(n)
    else:
        # Evidence חיצונית/אחרת – ננסה שדות סטנדרטיים אם קיימים
        for cand in [("p95_ms","perf.p95_ms"), ("score","ui.score"), ("passed","tests.passed"), ("rows","db.rows")]:
            src, dst = cand
            v = payload.get(src)
            if isinstance(v, bool): out[dst] = 1.0 if v else 0.0
            elif isinstance(v, (int,float)): out[dst] = float(v)
    return out

def _within(rule: MetricRule, a: float, b: float) -> bool:
    if math.isfinite(a) and math.isfinite(b):
        diff = abs(a - b)
        tol = rule.abs_tol + rule.rel_tol * max(1.0, abs(a), abs(b))
        return diff <= tol
    return False

def analyze_consistency(pv: ProvenanceStore, keys: List[str]) -> Dict[str,Any]:
    """
    מפיק מדדים מכל ראיה, משווה pairwise, מחשב ציון עקביות 0–100, ומחזיר פירוט סתירות.
    """
    measures: Dict[str, List[Tuple[str,float]]] = {}  # metric -> [(key, value)]
    recs: Dict[str, Dict[str,Any]] = {}

    for k in keys:
        rec = pv.get(k)
        if not rec: continue
        recs[k] = rec
        payload = rec.get("payload") or {}
        metrics = _extract_metrics(k, payload)
        for m, v in metrics.items():
            measures.setdefault(m, []).append((k, v))

    contradictions: List[Dict[str,Any]] = []
    total = 0
    agree = 0

    for m, pairs in measures.items():
        if len(pairs) <= 1:  # אין עם מי להשוות
            continue
        rule = Policy.get_rule(m)
        # השוואות pairwise
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                k1, v1 = pairs[i]; k2, v2 = pairs[j]
                total += 1
                if _within(rule, v1, v2):
                    agree += 1
                else:
                    contradictions.append({
                        "metric": m, "k1": k1, "v1": v1, "k2": k2, "v2": v2, "critical": rule.critical
                    })

    score = 100.0 if total == 0 else (100.0 * agree / total)
    ok = score >= Policy.min_consistency_score and not any(c["critical"] for c in contradictions)
    return {"ok": ok, "score": score, "contradictions": contradictions, "measures": measures}