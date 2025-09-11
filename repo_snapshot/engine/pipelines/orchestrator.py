# engine/pipelines/orchestrator.py
from __future__ import annotations
import asyncio, inspect, json
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Dict, List, Optional

from engine.pipeline_default import build_user_guarded
from engine.pipeline_events import emit_progress, emit_timeline
from engine.pipeline_respond_hook import pipeline_respond

# ---- קבלני-משנה (שכבות קיימות) שנעטוף:
from engine.prebuild.adapter_builder import ensure_capabilities
from engine.prebuild.tool_acquisition import ensure_tools
from engine.opt.optimizer import AutoOpt, kpi_to_reward

OPT_RUN = AutoOpt()
arms = [
  {"name":"vm", "x":[1,0,0,0]},
  {"name":"events", "x":[0,1,0,0]},
  {"name":"buildspec", "x":[0,0,1,0]}
]
# בחר runner לפי arm["name"]
i, arm = OPT_RUN.select(arms)

@dataclass
class Runner:
    name: str
    accepts: Callable[[Any], bool]
    run: Callable[[Any, Dict[str,Any]], Awaitable[Dict[str,Any]]]

class Orchestrator:
    def __init__(self, runners: List[Runner]) -> None:
        self.runners = runners

    async def run_any(self, spec: Any, ctx: Dict[str,Any]) -> Dict[str,Any]:
        from engine.recovery.freeze_window import is_frozen
        key = ctx.get("task") or (getattr(spec, "name", None) or ctx.get("spec_name") or "default")
        fr = is_frozen(str(key))
        if fr.get("frozen"):
            return {"ok": False, "stage":"frozen", "reason": fr.get("reason"), "until": fr.get("until")}

        from engine.recovery.backoff import allow
        bk = allow(str(key), attempts_max=int(ctx.get("__policy__",{}).get("attempts_max",2)))
        if not bk.get("ok"):
            return {"ok": False, "stage":"escalate", "attempts": bk.get("attempts")}
        user_id = ctx.get("user_id") or "anon"
        emit_timeline("orchestrator.start", f"user={user_id}")
        pre_missing = ensure_capabilities(spec, ctx)   # בונה stubs ליכולות חסרות (dry-run)
        pre_tools   = ensure_tools(spec, ctx)          # מתקין כלים עם evidence ורשיונות
        ctx.setdefault("__prebuild__", {}).update({"missing": pre_missing, "installed": pre_tools})
        # בוחרים Runner
        pick: Optional[Runner] = next((r for r in self.runners if r.accepts(spec)), None)
        if not pick:
            emit_timeline("orchestrator.error", "no_runner_match")
            raise ValueError("no_runner_for_spec")
        # עטיפת Strict-Grounded per-user
        guarded = await build_user_guarded(lambda s: self._run_instrumented(pick, s, ctx), user_id=user_id)
        out = await guarded(spec)

        # אחרי ריצה – בנה KPI מהריצה (latency/ok/cost):
        kpi = {"p95_ms": out.get("latency_ms", 1200.0), "error_rate": 0.0 if out.get("ok") else 1.0, "cost_usd": 0.0, "target_ms":1500.0}
        OPT_RUN.update(i, kpi_to_reward(kpi), context=arm["x"])
        from engine.recovery.backoff import clear
        clear(str(key))
        # אם יש טקסט/ארטיפקט — מעבירים דרך respond hook (אכיפת ראיות/מדיניות)
        if isinstance(out, dict) and ("text" in out or "pkg" in out):
            try:
                resp = pipeline_respond(ctx={**ctx, "__policy__": ctx.get("__policy__", {})},
                                        answer_text=out.get("text") or "")
                out = {**out, "respond": resp}
            except Exception as e:
                out = {**out, "respond_error": str(e)}
         # אחרי ריצה – בנה KPI מהריצה (latency/ok/cost):
        kpi = {"p95_ms": out.get("latency_ms", 1200.0), "error_rate": 0.0 if out.get("ok") else 1.0, "cost_usd": 0.0, "target_ms":1500.0}
        OPT_RUN.update(i, kpi_to_reward(kpi), context=arm["x"])
        emit_timeline("orchestrator.done", pick.name)
        from engine.recovery.backoff import clear
        if isinstance(out, dict) and out.get("ok"):
            clear(str(key))
        return out


    async def _run_instrumented(self, runner: Runner, spec: Any, ctx: Dict[str,Any]) -> Dict[str,Any]:
        emit_progress(0.0)
        emit_timeline("runner.start", runner.name)
        try:
            res = runner.run(spec, ctx)
            if inspect.isawaitable(res):
                res = await res
            emit_progress(100.0)
            return res  # כבר בתצורה הקנונית
        finally:
            emit_timeline("runner.end", runner.name)

# --------- עזרי זיהוי סוג קלט
def _is_vm_program(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(op, dict) and "op" in op for op in x)

def _is_json_spec_string(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    s = x.lstrip()
    if not (s.startswith("{") or s.startswith("[")):
        return False
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def _is_ab_spec(x: Any) -> bool:
    return isinstance(x, dict) and {"name","goal"} <= set(x.keys())

def _is_buildspec(x: Any) -> bool:
    try:
        from synth.specs import BuildSpec
        return isinstance(x, BuildSpec)
    except Exception:
        return False

# --------- בניית Runner-ים בפועל (ממופים ל-Shims)
from engine.pipelines.shims import run_vm, run_events_spec, run_ab_explore, run_buildspec, run_buildspec_multi

def default_runners() -> List[Runner]:
    return [
        Runner("vm", _is_vm_program, run_vm),
        Runner("events_spec", _is_json_spec_string, run_events_spec),
        Runner("ab_explore", _is_ab_spec, run_ab_explore),
        Runner("buildspec_full", _is_buildspec, run_buildspec),
        Runner("buildspec_multi", _is_buildspec, run_buildspec_multi),
    ]
