# imu_repo/engine/rollout_orchestrator.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Callable, Iterable, Optional
from engine.canary_controller import CanaryPlan, CanaryStage, CanaryRun
from engine.verify_bundle import verify_bundle, VerifyError
from engine.canary_autotune import suggest_next_percent
from engine.consistency_guard import check_drift_and_update

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
    stages: List[Dict[str,Any]],
    get_stage_claims: Optional[Callable[[str,int], List[Dict[str,Any]]]] = None,
    autotune: bool=False
) -> Dict[str,Any]:
    plan = CanaryPlan([CanaryStage(s["name"], int(s["percent"]), int(s.get("min_hold_sec",0))) for s in stages])
    run = CanaryRun(plan, backoff_base_sec=int(os.environ.get("IMU_CANARY_BACKOFF_BASE","5")))

    hist: List[Dict[str,Any]] = []
    _append_audit({"evt":"canary_start","stages":[(s.name,s.percent) for s in plan.stages]})

    while True:
        st = run.current()
        try:
            extra = get_stage_claims(st.name, st.percent) if get_stage_claims else None
            vb = verify_bundle(
                bundle=bundle, policy=policy, verifiers=verifiers,
                expected_scope=expected_scope, k=k, extra_kpi_claims=extra
            )
            headroom = float(vb["perf"]["headroom"]) if vb.get("perf") else 1.0
            near_miss = bool(vb["perf"]["near_miss"]) if vb.get("perf") else False

            # Consistency / Self-Heal
            chk = check_drift_and_update(extra or [], policy=policy, stage_name=st.name, percent=st.percent, near_miss=near_miss)
            heal = chk.get("heal")

            hist.append({"stage": st.name, "percent": st.percent, "gate":"pass",
                         "oks": vb.get("oks"), "perf_headroom": headroom, "near_miss": near_miss,
                         "drifts": chk.get("drifts")})
            _append_audit({"evt":"stage_pass","stage":st.name,"percent":st.percent,
                           "oks": vb.get("oks"), "perf_headroom": headroom, "near_miss": near_miss,
                           "drifts": chk.get("drifts")})

            # קידום/אדפטציה/ריפוי
            prev_idx = run.idx
            run.on_gate_pass()

            if heal:
                act = heal.get("action")
                if act == "rollback":
                    # חזרה אחורה של שלב ו/או הקטנת אחוז היעד
                    run.on_gate_fail(hard_abort=False)
                    if not run.completed and not run.aborted:
                        cur_pct = run.current().percent
                        new_pct = max(1, int(cur_pct * float(heal.get("rollback_factor",0.5))))
                        run.plan.stages[run.idx].percent = new_pct
                    _append_audit({"evt":"heal","mode":"rollback","reason":heal.get("reason")})
                elif act == "raise_require_fresh":
                    _append_audit({"evt":"heal","mode":"raise_require_fresh","reason":heal.get("reason")})
                    raise VerifyError(f"heal: require fresh evidence due to {heal.get('reason')}")
                else:  # freeze_autotune
                    autotune = False
                    _append_audit({"evt":"heal","mode":"freeze_autotune","reason":heal.get("reason")})

            if autotune and not run.completed:
                cur_pct = run.current().percent
                if near_miss:
                    suggested = max(1, min(100, cur_pct + 1))
                    run.plan.stages[run.idx].percent = suggested
                    _append_audit({"evt":"autotune","from":cur_pct,"to":suggested,"headroom":headroom,"mode":"near_miss_conservative"})
                else:
                    suggested = suggest_next_percent(cur_pct, headroom, policy)
                    run.plan.stages[run.idx].percent = suggested
                    _append_audit({"evt":"autotune","from":cur_pct,"to":suggested,"headroom":headroom,"mode":"adaptive"})

        except (Exception, VerifyError) as e:
            hist.append({"stage": st.name, "percent": st.percent, "gate":"fail", "err": str(e)})
            _append_audit({"evt":"stage_fail","stage":st.name,"percent":st.percent,"err":str(e)})
            run.on_gate_fail(hard_abort=False)

        if run.aborted or run.completed:
            break
        if not run.allow_advance():
            break

    return {"ok": True, "completed": run.completed, "aborted": run.aborted, "final_stage": run.current().name, "history": hist}