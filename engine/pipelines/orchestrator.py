# engine/pipelines/orchestrator.py
from __future__ import annotations
import asyncio, inspect, json
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Dict, List, Optional

from engine.pipeline_default import build_user_guarded
from engine.pipeline_events import emit_progress, emit_timeline
from engine.pipeline_respond_hook import pipeline_respond

# ---- קבלני-משנה (שכבות קיימות) שנעטוף:
from engine.pipeline import Engine as VMEngine              # (6)
from engine.synthesis_pipeline import SynthesisPipeline as _SynthV1  # (7) אם יש Class


@dataclass
class Runner:
    name: str
    accepts: Callable[[Any], bool]
    run: Callable[[Any, Dict[str,Any]], Awaitable[Dict[str,Any]]]

class Orchestrator:
    def __init__(self, runners: List[Runner]) -> None:
        self.runners = runners

    async def run_any(self, spec: Any, ctx: Dict[str,Any]) -> Dict[str,Any]:
        user_id = ctx.get("user_id") or "anon"
        emit_timeline("orchestrator.start", f"user={user_id}")
        # בוחרים Runner
        pick: Optional[Runner] = next((r for r in self.runners if r.accepts(spec)), None)
        if not pick:
            emit_timeline("orchestrator.error", "no_runner_match")
            raise ValueError("no_runner_for_spec")

        # עטיפת Strict-Grounded per-user
        guarded = await build_user_guarded(lambda s: self._run_instrumented(pick, s, ctx), user_id=user_id)
        out = await guarded(spec)

        # אם יש טקסט/ארטיפקט — מעבירים דרך respond hook (אכיפת ראיות/מדיניות)
        if isinstance(out, dict) and ("text" in out or "pkg" in out):
            try:
                resp = pipeline_respond(ctx={**ctx, "__policy__": ctx.get("__policy__", {})},
                                        answer_text=out.get("text") or "")
                out = {**out, "respond": resp}
            except Exception as e:
                out = {**out, "respond_error": str(e)}
        emit_timeline("orchestrator.done", pick.name)
        return out

    async def _run_instrumented(self, runner: Runner, spec: Any, ctx: Dict[str,Any]) -> Dict[str,Any]:
        emit_progress(0.0); emit_timeline("runner.start", runner.name)
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
    if not isinstance(x, str): return False
    s = x.lstrip()
    if not (s.startswith("{") or s.startswith("[")): return False
    try:
        json.loads(s); return True
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
from .shims import run_vm, run_events_spec, run_synth_v1, run_ab_explore, run_buildspec, run_buildspec_multi

def default_runners() -> List[Runner]:
    return [
        Runner("vm", _is_vm_program, run_vm),
        Runner("events_spec", _is_json_spec_string, run_events_spec),
        Runner("ab_explore", _is_ab_spec, run_ab_explore),
        Runner("buildspec_full", _is_buildspec, run_buildspec),
        Runner("buildspec_multi", _is_buildspec, run_buildspec_multi),
        Runner("synth_v1", _is_json_spec_string, run_synth_v1),
    ]
