# imu_repo/self_improve/apply.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from self_improve.fix_plan import FixPlan, FixAction
from self_improve.executors.ws_executor import WSExecutor
from self_improve.executors.guard_executor import GuardExecutor
from self_improve.executors.db_executor import DBExecutor

_EXECUTORS = [WSExecutor(), GuardExecutor(), DBExecutor()]

def _partition(actions: List[FixAction]) -> Dict[str, List[FixAction]]:
    parts: Dict[str, List[FixAction]] = {}
    for a in actions:
        dom = a.path[0] if a.path else "base"
        parts.setdefault(dom, []).append(a)
    return parts

def apply_with_executors(plans: List[FixPlan]) -> Dict[str,Any]:
    """
    מיישם את כל התוכניות עם מפעילים דומיין-ספציפיים, ומייצר בדיקות יחידה מתאימות.
    מחזיר תיאור מלא: אילו דומיינים שונו + אילו קבצי בדיקה נוצרו.
    """
    summary: Dict[str,Any] = {"domains":{}, "tests": []}
    for plan in plans:
        parts = _partition(plan.actions)
        for ex in _EXECUTORS:
            acts = parts.get(ex.domain) or []
            if not acts: continue
            res = ex.apply_actions(acts)
            summary["domains"][ex.domain] = res
            # generate tests
            for path, mod in ex.generate_tests(acts):
                summary["tests"].append({"path": path, "module": mod})
    return summary