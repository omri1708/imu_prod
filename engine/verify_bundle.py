# imu_repo/engine/verify_bundle.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Callable, List, Optional
from engine.rollout_quorum_gate import gate_release
from engine.perf_sla import enforce_perf_sla, PerfSlaError

class VerifyError(Exception): ...

def _claims_from_bundle(bundle: Dict[str,Any]) -> List[Dict[str,Any]]:
    # חיפוש claims נפוץ בחבילה חתומה (proof/package)
    for key in ("claims","evidence_claims","kpi_claims","metrics","body"):
        v = bundle.get(key)
        if isinstance(v, list) and all(isinstance(x, dict) for x in v):
            return v
    return []

def _nearmiss_threshold(policy: Dict[str,Any]) -> float:
    perf = (policy or {}).get("perf_sla") or {}
    nm = perf.get("near_miss_factor")
    try:
        th = float(nm)
        return th if th > 1.0 else 1.10
    except Exception:
        return 1.10  # בררת מחדל: 10%

def verify_bundle(
    *,
    bundle: Dict[str,Any],
    policy: Dict[str,Any],
    verifiers: Iterable[Callable[[Dict[str,Any],Dict[str,Any]], Dict[str,Any]]],
    expected_scope: str,
    k: int,
    extra_kpi_claims: Optional[List[Dict[str,Any]]] = None
) -> Dict[str,Any]:
    """
    מאחד:
      (1) gate_release → אימות חתימות/ראיות/טרסט
      (2) enforce_perf_sla → אכיפת SLA ביצועים
      (3) near-miss guard: headroom < threshold → ok אך מסומן כ-near_miss
    """
    # 1) אימות חתימות/ראיות/טרסט
    out = gate_release(bundle, policy, verifiers=verifiers, k=k, expected_scope=expected_scope)
    oks = int(out.get("oks", 0))
    if oks < k:
        raise VerifyError(f"quorum oks={oks} < required {k}")

    # 2) איסוף claims לביצועים
    claims = list(extra_kpi_claims or [])
    if not claims:
        claims = _claims_from_bundle(bundle)

    headroom = 1.0
    checked = []
    perf_ok = True
    perf_err: Optional[str] = None
    if claims:
        try:
            sla = enforce_perf_sla(claims, policy)
            headroom = float(sla.get("headroom", 1.0))
            checked = list(sla.get("checked") or [])
        except PerfSlaError as e:
            perf_ok = False
            perf_err = str(e)

    if not perf_ok:
        raise VerifyError(perf_err or "perf_sla breach")

    # 3) near-miss (עצירת האצה, לא כישלון)
    nm_thr = _nearmiss_threshold(policy)
    near_miss = (headroom < nm_thr)

    return {
        "ok": True,
        "oks": oks,
        "perf": {
            "headroom": headroom,
            "near_miss": near_miss,
            "checked": checked
        }
    }