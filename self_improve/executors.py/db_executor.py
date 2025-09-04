# imu_repo/self_improve/executors/db_executor.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from self_improve.executors.base import Executor
from self_improve.fix_plan import FixAction
from self_improve.patcher import apply_action
from engine.config import load_config, save_config
from self_improve.testgen import create_test

class DBExecutor(Executor):
    domain="db"

    def can_handle(self, action: FixAction) -> bool:
        return len(action.path)>=1 and action.path[0]=="db"

    def apply_actions(self, actions: List[FixAction]) -> Dict[str,Any]:
        cfg = load_config()
        if "db" not in cfg:
            cfg["db"] = {
                "sandbox": True,
                "max_conn": 8,
                "encrypt_at_rest": True
            }
        changed=False
        for a in actions:
            if not self.can_handle(a): continue
            apply_action(cfg, a); changed=True
        if changed:
            save_config(cfg)
        return {"changed": changed, "details": {"db": cfg.get("db", {})}}

    def generate_tests(self, actions: List[FixAction]) -> List[Tuple[str,str]]:
        code = """
        from __future__ import annotations
        from engine.config import load_config
        def run():
            cfg = load_config()
            db = cfg.get("db", {})
            assert db.get("sandbox") in (True, False)
            assert isinstance(db.get("max_conn", 0), int) and db["max_conn"]>0
            assert db.get("encrypt_at_rest") in (True, False)
            return True
        """
        return [create_test("db_exec_test", code)]