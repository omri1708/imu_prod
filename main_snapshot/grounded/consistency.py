# imu_repo/grounded/consistency.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

from grounded.provenance import ProvenanceStore
from grounded.contradiction_policy import policy_singleton as Policy, MetricRule
from provenance.provenance import aggregate_trust, evidence_expired


class ConsistencyError(Exception): ...
class MissingEvidence(ConsistencyError): ...
class ExpiredEvidence(ConsistencyError): ...
class LowTrust(ConsistencyError): ...
class NotEnoughSources(ConsistencyError): ...


def _match_evidences_for_binding(evs: List[Dict[str,Any]], url: str) -> List[Dict[str,Any]]:
    # התאמה לפי source_url; אם אין – נסה התאמות חלופיות (prefix בסיסי)
    matches = [e for e in evs if e.get("source_url","") == url]
    if matches: return matches
    # פרפיקס (למשל https://api.example/ ↔ https://api.example/v1/..)
    matches = [e for e in evs if url.startswith(str(e.get("source_url","")).rstrip("/"))]
    return matches

def check_ui_consistency(
    ui_claims: List[Dict[str,Any]],
    evidences: List[Dict[str,Any]],
    *,
    min_trust: float,
    min_sources: int
) -> Dict[str,Any]:
    """
    עבור כל claim של UI, דרוש לפחות מקור אחד שאינו פג-תוקף,
    Aggregate trust ≥ min_trust, ומספר מקורות ייחודיים ≥ min_sources (ברמת כל ה־UI).
    """
    if not ui_claims:
        return {"ok": True, "agg_trust": 1.0, "sources": 0, "checked": 0}

    # אסוף התאמות
    checked = 0
    all_matched: List[Dict[str,Any]] = []
    for c in ui_claims:
        url = c.get("source_url","")
        ms = _match_evidences_for_binding(evidences, url) if url else []
        if not ms:
            raise MissingEvidence(f"no evidence for binding {c.get('path')} -> {url}")
        fresh = [e for e in ms if not evidence_expired(e)]
        if not fresh:
            raise ExpiredEvidence(f"all evidences expired for {c.get('path')} -> {url}")
        all_matched.extend(fresh)
        checked += 1

    # מספר מקורות ייחודיים לכלל ה־UI
    uniq_sources = len({e.get("source_url","") for e in all_matched})
    if uniq_sources < int(min_sources):
        raise NotEnoughSources(f"need >= {min_sources} distinct sources, got {uniq_sources}")

    agg = aggregate_trust(all_matched)
    if agg < float(min_trust):
        raise LowTrust(f"agg_trust {agg:.2f} < min_trust {min_trust:.2f}")

    return {"ok": True, "agg_trust": agg, "sources": uniq_sources, "checked": checked}

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