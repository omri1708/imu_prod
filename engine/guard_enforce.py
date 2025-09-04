# imu_repo/engine/guard_enforce.py
from __future__ import annotations
import time
from typing import Dict, Any, List, Optional, Tuple
from grounded.provenance_confidence import is_fresh
from engine.errors import GuardRejection

def _eff_trust(ev: Dict[str, Any]) -> float:
    # אמון יעיל של ראיה = min(trust_evidence, trust_source)
    e_tr = float(ev.get("trust", 0.5))
    s_tr = float(ev.get("source_trust", 0.5))
    return min(max(0.0, e_tr), max(0.0, s_tr))

def _fresh_enough(ev: Dict[str, Any], max_age_s: Optional[float]) -> bool:
    if not is_fresh(ev):
        return False
    if max_age_s is None:
        return True
    now = time.time()
    ts = float(ev.get("ts", now))
    return (now - ts) <= float(max_age_s)

def _kinds_ok(evs: List[Dict[str,Any]], required_kinds: List[str] | None) -> Tuple[bool, List[str]]:
    if not required_kinds:
        return True, []
    kinds = {str(e.get("kind")) for e in evs}
    missing = [k for k in required_kinds if k not in kinds]
    return (len(missing) == 0), missing

def enforce_guard_before_respond(*, evidences: List[Dict[str,Any]], cfg: Dict[str,Any]) -> None:
    """
    משליך GuardRejection אם:
      - evidence.required=True ואין ראיות בכלל
      - אמון אפקטיבי לכל הראיות < min_trust
      - ראיות לא טריות (TTL) או חורגות מ-max_age_s
      - חסרים סוגי ראיות חובה (required_kinds)
      - min_count לא מושג
    """
    ev_cfg = dict(cfg.get("evidence", {}))
    guard_cfg = dict(cfg.get("guard", {}))

    required = bool(ev_cfg.get("required", True))
    min_trust = float(guard_cfg.get("min_trust", 0.7))
    max_age_s = guard_cfg.get("max_age_s", None)
    max_age_s = None if (max_age_s is None) else float(max_age_s)
    min_count = int(guard_cfg.get("min_count", 1))
    required_kinds = guard_cfg.get("required_kinds", None)
    if required_kinds is not None:
        required_kinds = [str(k) for k in required_kinds]

    if required and not evidences:
        raise GuardRejection("no_evidence", {"why":"required_evidence_missing"})

    # סינון ראיות לגיטימיות לפי טריות+סף אמון
    valids: List[Dict[str,Any]] = []
    too_old = 0
    too_low = 0
    for ev in evidences:
        fresh = _fresh_enough(ev, max_age_s)
        trust_ok = (_eff_trust(ev) >= min_trust)
        if fresh and trust_ok:
            valids.append(ev)
        else:
            if not fresh: too_old += 1
            if not trust_ok: too_low += 1

    if len(valids) < min_count:
        raise GuardRejection("insufficient_evidence", {
            "min_count": min_count, "have_valid": len(valids),
            "rejected_old": too_old, "rejected_low_trust": too_low,
            "min_trust": min_trust, "max_age_s": max_age_s
        })

    kinds_ok, missing = _kinds_ok(valids, required_kinds)
    if not kinds_ok:
        raise GuardRejection("missing_required_kinds", {"missing": missing})
