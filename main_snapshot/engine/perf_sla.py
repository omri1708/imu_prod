# imu_repo/engine/perf_sla.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

class PerfSlaError(Exception): ...

def _f(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _norm_metric_key(claim: Dict[str,Any]) -> Optional[str]:
    """
    מייצר מזהה "מנורמל" להשוואה למדיניות:
      latency p95 → "latency_ms.p95"
      latency p99 → "latency_ms.p99"
      error rate  → "error_rate"
      throughput  → "throughput_rps"
    """
    t = str(claim.get("type","")).lower()
    unit = str((claim.get("schema") or {}).get("unit","")).lower()
    quant = str(claim.get("quantile","")).lower()
    if t == "latency" and unit == "ms":
        if quant in ("p95","p99"):
            return f"latency_ms.{quant}"
        # אם לא סופק quantile, נחשיב כ-p95 בררת מחדל:
        return "latency_ms.p95"
    if t in ("error_rate","errors","errorrate"):
        return "error_rate"
    # "kpi" עם יחידה rps/tps → throughput
    if t in ("kpi","throughput","tps","rps"):
        if unit in ("rps","tps"):
            return "throughput_rps"
        # נפוצה גם כטקסט "tps=..."
        text = str(claim.get("text","")).lower()
        if "tps=" in text or "rps=" in text:
            return "throughput_rps"
    return None

def _policy_thresholds(policy: Dict[str,Any]) -> Dict[str, Dict[str,float]]:
    """
    צפוי ב-policy:
    "perf_sla": {
      "latency_ms": {"p95_max": 150.0, "p99_max": 300.0},
      "throughput_rps": {"min": 100.0},
      "error_rate": {"max": 0.01}
    }
    """
    perf = (policy or {}).get("perf_sla") or {}
    out: Dict[str,Dict[str,float]] = {}
    # latency
    lat = perf.get("latency_ms") or {}
    l95 = _f(lat.get("p95_max"))
    l99 = _f(lat.get("p99_max"))
    if l95 is not None: out["latency_ms.p95"] = {"max": l95}
    if l99 is not None: out["latency_ms.p99"] = {"max": l99}
    # throughput
    thr = perf.get("throughput_rps") or {}
    thr_min = _f(thr.get("min"))
    if thr_min is not None: out["throughput_rps"] = {"min": thr_min}
    # error rate
    er = perf.get("error_rate") or {}
    er_max = _f(er.get("max"))
    if er_max is not None: out["error_rate"] = {"max": er_max}
    return out

def enforce_perf_sla(claims: List[Dict[str,Any]], policy: Dict[str,Any]) -> Dict[str,Any]:
    """
    עובר על claims, מחלץ ערכים רלוונטיים ומאכף מול ספי ה-policy.
    אם יש חריגה—זורק PerfSlaError. מחזיר גם "headroom" מנורמל (כמה מרווח נשאר).
    """
    th = _policy_thresholds(policy)
    if not th:
        return {"ok": True, "headroom": 1.0, "checked": []}

    checked = []
    worst_headroom = float("inf")  # קטן יותר = רע יותר; <1 → חריגה
    for c in (claims or []):
        key = _norm_metric_key(c)
        if not key or key not in th:
            continue
        val = _f(c.get("value"))
        if val is None:
            # ניסיון חילוץ מטקסט "tps=123"
            txt = str(c.get("text","")).lower()
            import re
            if key == "throughput_rps":
                m = re.search(r"(?:tps|rps)\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt)
                if m: val = _f(m.group(1))
        if val is None:
            # אין ערך מדיד—מדלגים (לא יכביד על החישוב)
            continue

        lim = th[key]
        if "max" in lim:
            limit = lim["max"]
            headroom = (limit / val) if val > 0 else float("inf")
            checked.append((key, val, f"<= {limit}"))
            worst_headroom = min(worst_headroom, headroom)
            if val > limit:
                raise PerfSlaError(f"SLA breach: {key}={val} > max {limit}")
        elif "min" in lim:
            limit = lim["min"]
            headroom = (val / limit) if limit > 0 else float("inf")
            checked.append((key, val, f">= {limit}"))
            worst_headroom = min(worst_headroom, headroom)
            if val < limit:
                raise PerfSlaError(f"SLA breach: {key}={val} < min {limit}")

    if worst_headroom == float("inf"):
        worst_headroom = 1.0
    return {"ok": True, "headroom": float(worst_headroom), "checked": checked}
