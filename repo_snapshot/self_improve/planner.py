# imu_repo/self_improve/planner.py
from __future__ import annotations
from typing import Dict, Any, List
from self_improve.fix_plan import FixPlan, FixAction

def _set(path: List[str], val: Any)->FixAction: return FixAction(path,"set",val)
def _dec(path: List[str], val: Any)->FixAction: return FixAction(path,"dec",val)
def _inc(path: List[str], val: Any)->FixAction: return FixAction(path,"inc",val)

def plan_from_stats(stats: Dict[str,Any])->List[FixPlan]:
    """
    יוצר(ות) FixPlan לפי חריגות בנתונים: latency/error/gate-denied/throughput.
    חוקים דטרמיניסטיים (ללא LLM).
    """
    out: List[FixPlan] = []
    lat = stats.get("latency", {}) or {}
    p95 = float(lat.get("p95_ms") or 0.0)
    err = float(stats.get("error_rate", 0.0))
    gate = float(stats.get("gate_denied_rate", 0.0))
    thr = float(stats.get("throughput_rps", 0.0))

    # אם p95 גבוה: הקטן chunk_size, הקטן max_pending, ודא דחיסה מופעלת
    if p95 > 80.0:
        out.append(FixPlan(
            reason="p95_high",
            actions=[
                _set(["ws","permessage_deflate"], True),
                _dec(["ws","chunk_size"], 16000),     # הורדה של 16KB
                _dec(["ws","max_pending_msgs"], 256), # הפחתת לחץ זיכרון/תורים
            ],
            notes="Reduce WebSocket payloads, enforce deflate, reduce pending queue to curb tail latency.",
            expected_effect={"p95_ms": "drop ~10-30%"}
        ))

    # אם שיעור כשלים גבוה: העלה אמינות ראיות (להוריד כשלים עקב gate), אך לא חמור מדי
    if err > 0.02:
        out.append(FixPlan(
            reason="error_rate_high",
            actions=[
                _inc(["guard","min_trust"], 0.05),     # עלה את רף האמון בדרישת ראיות
                _set(["guard","max_age_s"], 1800),     # הקשחת טריות
            ],
            notes="Tighten evidence trust/age to avoid flaky paths; reduce handler fall-through.",
            expected_effect={"error_rate": "drop"}
        ))

    # אם gate_denied גבוה: איזון — ייתכן שהרף גבוה מדי → הורד מעט
    if gate > 0.02 and err <= 0.02:
        out.append(FixPlan(
            reason="gate_denied_high",
            actions=[
                _dec(["guard","min_trust"], 0.05),
                _inc(["guard","max_age_s"], 900),
            ],
            notes="Balance Gate sensitivity to reduce denials without harming overall reliability.",
            expected_effect={"gate_denied_rate": "drop"}
        ))

    # אם throughput נמוך: העלה max_pending (בזהירות)
    if thr < 1e-3:  # למשל אין תנועה — אין מה לשנות
        ...
    elif thr < 0.5:
        out.append(FixPlan(
            reason="throughput_low",
            actions=[
                _inc(["ws","max_pending_msgs"], 256),
            ],
            notes="Increase pending window to improve pipeline throughput.",
            expected_effect={"throughput_rps": "rise"}
        ))

    return out