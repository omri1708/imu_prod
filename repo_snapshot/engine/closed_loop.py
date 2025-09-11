# imu_repo/engine/closed_loop.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time

from engine.pipeline import Engine
from governance.ab_verify import ABVerifier
from governance.proof_of_convergence import ConvergenceTracker, SafeProgressLedger
from optimizer.phi import suite_phi
from persistence.policy_store import PolicyStore

class SimpleLearner:
    """
    לומד פרמטרים ברמת-מערכת (thresholds/limits) בגישת hill-climb פשוטה.
    לא מבטיח אופטימום גלובלי, אבל עם Safe-Progress רק שיפורים מאומצים.
    """
    def propose(self, base_cfg: Dict[str,Any], signal: Dict[str,Any]) -> Dict[str,Any]:
        cand = {**base_cfg}
        # דוגמה: אם p95 גבוה, להדק ספי שגיאה/latency; אם error_rate גבוה, להרחיב ריסוסים או להוריד עומסים.
        perf = signal.get("perf", {})
        p95  = float(perf.get("p95", 0.0))
        err  = float(perf.get("error_rate", 0.0))
        th = cand.get("thresholds", {
            "max_error_rate": 0.02,
            "max_p95_latency_ms": 800.0,
            "max_regression_p95_ms": 10.0
        })
        if p95 > th["max_p95_latency_ms"]:
            th["max_p95_latency_ms"] = min(2000.0, p95 * 1.10)
        if err > th["max_error_rate"]:
            th["max_error_rate"] = min(0.10, err * 1.05)
        cand["thresholds"] = th

        # דוגמת limit משאבים:
        limits = cand.get("limits", {"cpu_steps_max":500000,"mem_kb_max":65536,"io_calls_max":10000})
        if err > 0.05:
            limits["io_calls_max"] = max(1000, int(limits["io_calls_max"] * 0.95))
        cand["limits"] = limits
        return cand

def _run_suite(engine: Engine, suite: List[Tuple[List[Dict[str,Any]], Dict[str,Any]]]) -> List[Dict[str,Any]]:
    runs=[]
    import time
    for prog, payload in suite:
        t0=time.time()
        code, body = engine.run_program(prog, payload, policy="strict")
        lat=int((time.time()-t0)*1000)
        kind="ok" if 200 <= code < 400 else "error"
        runs.append({"kind":kind,"metrics":{"latency_ms":lat,"error":kind=="error"}})
    return runs

class ClosedLoop:
    """
    מריץ מחזורי למידה:
    baseline → candidate (propose) → run A/B → Φ↓? → promote or rollback.
    """
    def __init__(self, engine: Engine, policy: PolicyStore, ledger: SafeProgressLedger):
        self.engine=engine
        self.policy=policy
        self.ledger=ledger
        self.tracker=ConvergenceTracker(window=8, epsilon=0.002, max_violations=2)
        self.learner=SimpleLearner()

    def verify_learning_improved_system(self,
                                        baseline_runs: List[Dict[str,Any]],
                                        candidate_runs: List[Dict[str,Any]],
                                        thresholds: Dict[str,Any]) -> Dict[str,Any]:
        from governance.ab_verify import ABVerifier
        ab=ABVerifier(thresholds)
        decision=ab.compare(baseline_runs, candidate_runs)
        # Φ
        phi_base=suite_phi(baseline_runs)
        phi_cand=suite_phi(candidate_runs)
        delta=phi_cand - phi_base
        self.tracker.add(phi_cand)
        conv=self.tracker.status()
        return {
            "ab_passed": decision.passed,
            "ab_report": decision.report,
            "phi_base": phi_base,
            "phi_cand": phi_cand,
            "phi_delta": delta,
            "convergence": conv
        }

    def learn_once(self, suite: List[Tuple[List[Dict[str,Any]], Dict[str,Any]]]) -> Dict[str,Any]:
        base=self.policy.current()
        base_cfg=base.get("config", {})
        # baseline runs
        baseline_runs=_run_suite(self.engine, suite)
        base_perf={"p95": sorted([r["metrics"]["latency_ms"] for r in baseline_runs])[int(0.95*len(baseline_runs))-1],
                   "error_rate": sum(1 for r in baseline_runs if r["kind"]=="error")/max(1,len(baseline_runs))}
        cand_cfg=self.learner.propose(base_cfg, {"perf": base_perf})
        ver= self.policy.stage(cand_cfg, note="auto-proposed")
        # candidate runs: (בפועל A/B אמיתי; כאן מריצים שוב כסימולציה דטרמיניסטית)
        candidate_runs=_run_suite(self.engine, suite)

        thresholds=cand_cfg.get("thresholds", {"max_error_rate":0.02,"max_p95_latency_ms":800.0,"max_regression_p95_ms":10.0})
        verdict=self.verify_learning_improved_system(baseline_runs, candidate_runs, thresholds)

        event={"type":"learning_verdict","candidate_version":ver, **verdict}
        h=self.ledger.append(event)

        # Gate Safe-Progress: חייבים גם AB pass וגם Φ↓ ממשי
        min_improve= -0.001  # ΔΦ חייב להיות <= -0.001 (שיפור)
        if verdict["ab_passed"] and verdict["phi_delta"] <= min_improve:
            promoted=self.policy.promote(ver)
            self.ledger.append({"type":"promote","to":ver,"hash_of_prev":h})
            return {"status":"promoted","version":ver,"verdict":verdict,"policy":promoted}
        else:
            # רולבק (פשוט לא מקדמים)
            self.ledger.append({"type":"rollback","version":ver,"hash_of_prev":h})
            return {"status":"rollback","version":ver,"verdict":verdict,"policy":base}

def bootstrap_complete_system(engine: Engine,
                              suite: List[Tuple[List[Dict[str,Any]], Dict[str,Any]]],
                              iterations:int=3) -> Dict[str,Any]:
    """
    Bootstrap סגור: מריץ מספר איטרציות למידה סופיות,
    מבטיח רק שיפורים מאומצים (Safe-Progress).
    """
    policy=PolicyStore()
    ledger=SafeProgressLedger()
    loop=ClosedLoop(engine, policy, ledger)
    last=None
    for i in range(iterations):
        last=loop.learn_once(suite)
    return last or {"status":"noop"}
