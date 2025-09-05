# imu_repo/engine/synthesis_pipeline.py
from __future__ import annotations
from typing import Any, Dict, Callable, List, Optional
import time
from contextlib import contextmanager

from perf.p95 import P95Tracker
from engine.quarantine import CapabilityGuard, Quarantined
from engine.alerts import alert
from engine.reputation import ReputationRegistry
from engine.policy_overrides import apply_user_overrides
from engine.respond_strict import RespondStrict
from engine.verify_bundle import verify_bundle
from engine.rollout_orchestrator import run_canary_orchestration
from engine.contracts_gate import enforce_respond_contract

from governance.user_policy import get_user_policy
from audit.log import AppendOnlyAudit
from governance.slo_gate import gate_p95
from perf.monitor import monitor_global

from synth.specs import parse_spec
from synth.plan import build_plan
from synth.generate import generate_artifacts
from synth.test import run_tests
from synth.verify import verify_artifacts
from synth.package import package_release
from synth.canary import shadow_and_canary
from synth.rollout import gated_rollout


class PipelineError(Exception): ...

@contextmanager
def _timed(tracker: P95Tracker):
    t0 = time.time()
    try:
        yield
    finally:
        dt_ms = (time.time() - t0) * 1000.0
        tracker.add(dt_ms)

class SynthesisPipeline:
    """
    פייפליין גנרי: spec -> plan -> generate -> test -> verify -> package -> canary -> rollout
    מוסיף:
      - p95 SLO gates לפי policy
      - בידוד יכולות (quarantine)
      - Reputation כ-hook למדיניות (למשל ב-gates אחרים במערכת)
    """
    def __init__(self, *, base_policy: Dict[str,Any], http_fetcher=None, sign_key_id: Optional[str]="root", now=None):
        self.policy = dict(base_policy or {})
        self._now = now or (lambda: time.time())
        self.responder = RespondStrict(base_policy=base_policy, http_fetcher=http_fetcher, sign_key_id=sign_key_id)
        self.trackers: Dict[str, P95Tracker] = {
            "plan": P95Tracker(window=int(self.policy.get("p95_window", 500))),
            "generate": P95Tracker(window=int(self.policy.get("p95_window", 500))),
            "test": P95Tracker(window=int(self.policy.get("p95_window", 500))),
            "verify": P95Tracker(window=int(self.policy.get("p95_window", 500))),
            "package": P95Tracker(window=int(self.policy.get("p95_window", 500))),
            "canary": P95Tracker(window=int(self.policy.get("p95_window", 500))),
            "rollout": P95Tracker(window=int(self.policy.get("p95_window", 500)))
        }
        self.guard = CapabilityGuard(now=self._now)
        self.reputation = self.policy.get("reputation")
        
        self.responder = RespondStrict(base_policy=base_policy, http_fetcher=http_fetcher, sign_key_id=sign_key_id)
        if self.reputation is None:
            self.reputation = ReputationRegistry()
            self.policy["reputation"] = self.reputation

        self.step_impls: Dict[str, Callable[[Dict[str,Any]], Dict[str,Any]]] = {}

    def register(self, step: str, fn: Callable[[Dict[str,Any]], Dict[str,Any]]) -> None:
        self.step_impls[step] = fn

    def _slo_gate(self, step: str) -> None:
        p95_limit = self.policy.get(f"{step}_p95_ms")
        if p95_limit is None:
            return
        p95 = self.trackers[step].p95()
        if p95 > float(p95_limit):
            alert("ERROR", "SLO p95 breach", {"step": step, "p95": p95, "limit": p95_limit})
            raise PipelineError(f"slo_breach:{step}: p95={p95:.2f}ms > limit={p95_limit}ms")

    def _call_with_quarantine(self, cap: str, fn: Callable[[Dict[str,Any]], Dict[str,Any]], ctx: Dict[str,Any]) -> Dict[str,Any]:
        try:
            self.guard.before_call(cap)
            out = fn(ctx)
            ok = bool(out.get("ok", True))
            vio = int(out.get("_violations", 0))
            self.guard.after_call(cap, ok=ok, violations=vio, policy=self.policy)
            if not ok:
                raise PipelineError(f"{cap}_failed")
            # reputation hook: הצלחה -> משפר מוניטין של המקור (אם יש)
            src = out.get("_source_id")
            if isinstance(src, str) and src:
                self.reputation.update_on_success(src)
            return out
        except Quarantined as q:
            alert("WARNING", "cap_quarantined", {"cap": cap, "reason": str(q)})
            raise
        except Exception as e:
            # כישלון → עדכון מוניטין של המקור אם יש
            src = ctx.get("_source_id")
            if isinstance(src, str) and src:
                self.reputation.update_on_violation(src)
            self.guard.after_call(cap, ok=False, violations=1, policy=self.policy)
            raise

    def run(self, ctx: Dict[str,Any]) -> Dict[str,Any]:
        # סדר קבוע, אך כל שלב רשום רק אם יש מימוש
        for step in ["plan","generate","test","verify","package","canary","rollout"]:
            fn = self.step_impls.get(step)
            if not fn:
                continue
            with _timed(self.trackers[step]):
                out = self._call_with_quarantine(step, fn, ctx)
                ctx.update({f"{step}_out": out})
            # SLO gate אחרי כל שלב
            self._slo_gate(step)
        return {"ok": True, "ctx": ctx, "p95": {k: v.p95() for k,v in self.trackers.items()}}

    
    def run_once(self,
                *,
                ctx: Dict[str,Any],
                generate_fn: Callable[[Dict[str,Any]], tuple],
                verifiers: List[Callable[[Dict[str,Any],Dict[str,Any]], Dict[str,Any]]],
                rollout_stages: List[Dict[str,Any]],
                expected_scope: str = "deploy",
                k: int = 1,
                autotune: bool = False,
                get_stage_claims: Optional[Callable[[str,int], List[Dict[str,Any]]]] = None
                ) -> Dict[str,Any]:
        # 1) יצירה/סינתזה של תשובה+חבילה חתומה
        out = self.responder.respond(ctx=ctx, generate=generate_fn)
        bundle = out["bundle"]; eff_policy = out["policy"]

        # 2) אימות bundle
        vouts = [v(bundle, eff_policy) for v in verifiers]
        oks = [vo.get("ok") for vo in vouts]
        if not all(oks):
            return {"ok": False, "stage":"verify", "errors": vouts}

        # 3) rollout תזמורי + Consistency/Self-Heal (חובר בשלב 101)
        roll = run_canary_orchestration(
            bundle=bundle, policy=eff_policy, verifiers=verifiers, expected_scope=expected_scope,
            k=k, stages=rollout_stages, get_stage_claims=get_stage_claims, autotune=autotune
        )
        if not roll.get("ok"):
            return {"ok": False, "stage":"rollout", "details": roll}
        return {"ok": True, "bundle": bundle, "rollout": roll, "text": out["text"], "policy": eff_policy}
    

AUDIT = AppendOnlyAudit("var/audit/pipeline.jsonl")


def run_pipeline(user: str, spec_text: str) -> Dict[str, Any]:
 
    t0 = time.time()
    policy, ev_index = get_user_policy(user)

    spec = parse_spec(spec_text)
    AUDIT.append({"stage":"parse","user":user,"ok":True})

    plan = build_plan(spec)
    AUDIT.append({"stage":"plan","user":user,"ok":True})

    artifacts, claims, evidence = generate_artifacts(plan)  # מייצר גם claims/evidence
    AUDIT.append({"stage":"generate","user":user,"claims":len(claims),"evidence":len(evidence)})

    # מוודא שה-Evidence עומד במדיניות המשתמש לפני המשך
    from engine.contracts_gate import enforce_respond_contract
    enforce_respond_contract("pipeline_generate", claims, evidence, policy, ev_index)

    tests_ok = run_tests(artifacts)
    AUDIT.append({"stage":"test","user":user,"ok":tests_ok})
    if not tests_ok:
        return {"ok": False, "stage":"test"}

    verified = verify_artifacts(artifacts, claims, evidence)
    AUDIT.append({"stage":"verify","user":user,"ok":verified})
    if not verified:
        return {"ok": False, "stage":"verify"}

    pkg_path = package_release(artifacts)
    AUDIT.append({"stage":"package","user":user,"pkg":pkg_path})

    # Shadow/Canary אוספים KPIs ובוחנים מול baseline
    canary_ok = shadow_and_canary(pkg_path, policy=policy)
    AUDIT.append({"stage":"canary","user":user,"ok":canary_ok})
    if not canary_ok:
        return {"ok": False, "stage":"canary"}

    # כבודק ביצועים בזמן אמת (p95)
    elapsed_ms = (time.time() - t0) * 1000.0
    monitor_global.observe_ms(elapsed_ms)
    gate_p95(max_ms=300.0)

    rollout_ok = gated_rollout(pkg_path, policy=policy)
    AUDIT.append({"stage":"rollout","user":user,"ok":rollout_ok,"elapsed_ms":elapsed_ms})
    return {"ok": rollout_ok, "pkg": pkg_path, "latency_ms": elapsed_ms}