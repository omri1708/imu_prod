# imu_repo/self_improve/executors/guard_executor.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Awaitable
import asyncio
from self_improve.executors.base import Executor
from self_improve.fix_plan import FixAction
from self_improve.patcher import apply_action
from engine.config import load_config, save_config
from self_improve.testgen import create_test

class GuardExecutor(Executor):
    domain="guard"

    def can_handle(self, action: FixAction) -> bool:
        return len(action.path)>=1 and action.path[0]=="guard"

    def apply_actions(self, actions: List[FixAction]) -> Dict[str,Any]:
        cfg = load_config()
        changed=False
        for a in actions:
            if not self.can_handle(a): continue
            apply_action(cfg, a); changed=True
        if changed:
            save_config(cfg)
        return {"changed": changed, "details": {"guard": cfg.get("guard", {})}}

    def generate_tests(self, actions: List[FixAction]) -> List[Tuple[str,str]]:
        # הטסט: בודק שבלי ראיות — gate ננעל; עם ראיה מעל min_trust — עובר.
        code = r'''
        from __future__ import annotations
        import asyncio
        from engine.config import load_config
        from engine.evidence_middleware import guarded_handler
        from grounded.claims import current

        async def _noop(x:str)->str:
            # ה-handler עצמו רק מחזיר טקסט; ה-gate דואג להוכחות
            return f"ok:{x}"

        def run():
            cfg = load_config()
            min_trust = float(cfg.get("guard",{{}}).get("min_trust", 0.7))
            async def scenario():
                safe = await guarded_handler(_noop, min_trust=min_trust)
                # 1) בלי ראיות — צריך להיכשל
                failed=False
                try:
                    await safe("x")
                except Exception as e:
                    failed=True
                assert failed, "gate should deny without evidences"

                # 2) עם ראיה מתאימה — צריך לעבור
                cur = current()
                cur.add_evidence("t1", {{"source_url":"https://example","trust": min_trust+0.05, "ttl_s":60}})
                out = await safe("y")
                assert isinstance(out, dict) and out.get("text")=="ok:y" and out.get("claims"), "guarded response must carry claims"
                return True
            return asyncio.get_event_loop().run_until_complete(scenario())
        '''
        return [create_test("guard_exec_test", code)]