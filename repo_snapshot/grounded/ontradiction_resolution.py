from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import math
from grounded.provenance import ProvenanceStore
from grounded.contradiction_policy import policy_singleton as Policy, MetricRule
from grounded.consistency import _extract_metrics, _within

@dataclass
class ResolutionResult:
    ok: bool
    reason: str
    effective: Dict[str, float]              # ערכים שהוכרעו
    used: Dict[str, List[Tuple[str, float]]] # metric -> [(key, val)] ששימשו בהחלטה
    dropped: Dict[str, List[Tuple[str, float, float]]]  # metric -> [(key, val, trust)] שנפסלו כאאוטלייר/נמוך־אמון
    proof: Dict[str, Any]                    # סיכום הוכחה (ספים, משקלים, סטטיסטיקות)

def _wmean(items: List[Tuple[float, float]]) -> float:
    # items: [(value, weight)]
    num = sum(v*w for v, w in items)
    den = sum(w for _, w in items)
    return num / den if den > 0 else float("nan")

def _wmedian(items: List[Tuple[float, float]]) -> float:
    # חתך משוקלל פשוט
    if not items: return float("nan")
    items = sorted(items, key=lambda t: t[0])
    total_w = sum(w for _,w in items)
    acc = 0.0
    mid = total_w / 2.0
    for v, w in items:
        acc += w
        if acc >= mid:
            return v
    return items[-1][0]

def resolve_contradictions(pv: ProvenanceStore, keys: List[str], *,
                           trust_cut: float = 0.75,
                           method: str = "wmedian") -> ResolutionResult:
    """
    מסיר ראיות נמוכות־אמון (trust < trust_cut), מחשב ערך אפקטיבי לכל metric (wmedian/wmean),
    ובודק אם המטריקות שנותרו עקביות לפי הכללים. אם כן — מחזיר ok=True והוכחה.
    """
    measures: Dict[str, List[Tuple[str, float, float]]] = {}  # metric -> [(key, value, trust)]
    for k in keys:
        rec = pv.get(k)
        if not rec: continue
        payload = rec.get("payload") or {}
        trust = float(rec.get("trust", 0.0))
        for m, v in _extract_metrics(k, payload).items():
            measures.setdefault(m, []).append((k, float(v), trust))

    effective: Dict[str, float] = {}
    used: Dict[str, List[Tuple[str,float]]] = {}
    dropped: Dict[str, List[Tuple[str,float,float]]] = {}
    proof: Dict[str, Any] = {"trust_cut": trust_cut, "method": method, "metrics": {}}

    # 1) פילטר לפי אמון
    filtered: Dict[str, List[Tuple[str, float, float]]] = {}
    for m, triples in measures.items():
        kept, drp = [], []
        for k, v, t in triples:
            (kept if t >= trust_cut else drp).append((k, v, t))
        filtered[m] = kept
        dropped[m] = drp

    # 2) איחוד משוקלל
    for m, triples in filtered.items():
        if not triples:
            continue
        rule: MetricRule = Policy.get_rule(m)
        arr = [(v, t) for _, v, t in triples]
        eff = _wmedian(arr) if method == "wmedian" else _wmean(arr)
        effective[m] = eff
        used[m] = [(k, v) for k, v, _ in triples]
        proof["metrics"][m] = {
            "rule": {"rel_tol": rule.rel_tol, "abs_tol": rule.abs_tol, "critical": rule.critical},
            "values": [{"key": k, "val": v, "trust": t} for k, v, t in triples],
            "effective": eff
        }

    # 3) בדיקת עקביות פנימית אחרי חיתוך
    total = 0; agree = 0; contradictions = []
    for m, triples in filtered.items():
        if len(triples) <= 1: 
            continue
        rule = Policy.get_rule(m)
        for i in range(len(triples)):
            for j in range(i+1, len(triples)):
                k1, v1, _ = triples[i]; k2, v2, _ = triples[j]
                total += 1
                if _within(rule, v1, v2):
                    agree += 1
                else:
                    contradictions.append({"metric": m, "k1": k1, "v1": v1, "k2": k2, "v2": v2, "critical": rule.critical})

    score = 100.0 if total==0 else (100.0 * agree / total)
    ok = (score >= Policy.min_consistency_score) and not any(c["critical"] for c in contradictions)
    proof["consistency_score"] = score
    proof["contradictions_after_cut"] = contradictions
    return ResolutionResult(ok=ok, reason=("ok" if ok else "after_cut_inconsistent"),
                            effective=effective, used=used, dropped=dropped, proof=proof)