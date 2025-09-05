# imu_repo/engine/rollout_orchestrator.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Callable, Iterable
from engine.canary_controller import CanaryPlan, CanaryStage, CanaryRun
from engine.rollout_quorum_gate import gate_release

def _audit_dir() -> str:
    d = os.environ.get("IMU_AUDIT_DIR") or ".audit"
    os.makedirs(d, exist_ok=True); return d

def _append_audit(row: Dict[str,Any], fname: str="rollout_orchestrator.jsonl") -> None:
    p = os.path.join(_audit_dir(), fname)
    with open(p, "a", encoding="utf-8") as f:
        row = {"ts": time.time(), **row}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def run_canary_orchestration(
    *,
    bundle: Dict[str,Any],
    policy: Dict[str,Any],
    verifiers: Iterable[Callable[[Dict[str,Any],Dict[str,Any]], Dict[str,Any]]],
    expected_scope: str,
    k: int,
    stages: List[Dict[str,Any]]
) -> Dict[str,Any]:
    """
    stages: [{"name":"5%","percent":5,"min_hold_sec":0}, ...]
    מחזיר {"ok":True,"completed":bool,"final_stage":{...},"history":[...]} או זורק שגיאה מן השער.
    """
    plan = CanaryPlan([CanaryStage(s["name"], int(s["percent"]), int(s.get("min_hold_sec",0))) for s in stages])
    run = CanaryRun(plan, backoff_base_sec= int(os.environ.get("IMU_CANARY_BACKOFF_BASE","5")))

    hist: List[Dict[str,Any]] = []
    _append_audit({"evt":"canary_start","stages":[(s.name,s.percent) for s in plan.stages]})

    while True:
        st = run.current()
        try:
            out = gate_release(bundle, policy, verifiers=verifiers, k=k, expected_scope=expected_scope)
            hist.append({"stage": st.name, "percent": st.percent, "gate":"pass", "oks": out.get("oks"), "total": out.get("total")})
            _append_audit({"evt":"stage_pass","stage":st.name,"percent":st.percent,"oks":out.get("oks"),"total":out.get("total")})
            run.on_gate_pass()
        except Exception as e:
            hist.append({"stage": st.name, "percent": st.percent, "gate":"fail", "err": str(e)})
            _append_audit({"evt":"stage_fail","stage":st.name,"percent":st.percent,"err":str(e)})
            run.on_gate_fail(hard_abort=False)

        if run.aborted or run.completed:
            break
        # אם ה-hold העתידי עדיין לא הגיע (בגלל backoff), נשאיר את הלולאה להסתובב “לוגית”
        # בפועל מערכת ריצה אמיתית תזמן טסק עתידי; כאן נשאר סינכרוני ובודקים אם הזמן חלף.
        if not run.allow_advance():
            # מצב “מחכים” — נשבור כדי לא להיתקע; Responsibility של caller לקרוא שוב.
            break

    return {"ok": True, "completed": run.completed, "aborted": run.aborted, "final_stage": run.current().name, "history": hist}