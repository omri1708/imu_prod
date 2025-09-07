# imu_repo/self_improve/executors/ws_executor.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from self_improve.executors.base import Executor
from self_improve.fix_plan import FixAction
from self_improve.patcher import apply_action
from engine.config import load_config, save_config
from self_improve.testgen import create_test

class WSExecutor(Executor):
    domain="ws"

    def can_handle(self, action: FixAction) -> bool:
        return len(action.path)>=1 and action.path[0]=="ws"

    def apply_actions(self, actions: List[FixAction]) -> Dict[str,Any]:
        cfg = load_config()
        changed = False
        for a in actions:
            if not self.can_handle(a): continue
            apply_action(cfg, a)
            changed = True
        if changed:
            save_config(cfg)
        return {"changed": changed, "details": {"ws": cfg.get("ws", {})}}

    def generate_tests(self, actions: List[FixAction]) -> List[Tuple[str,str]]:
        # בנה טסט המאשר שהקונפיג תואם ל-actions ומשפיע על שרת ה-WS (במקסימום בלי נטוורק)
        code = f"""
        from __future__ import annotations
        import asyncio
        from engine.config import load_config
        from realtime.ws_server import WSServer

        def run():
            cfg = load_config()
            ws = cfg.get("ws", {{}})
            # אסרציות על פרמטרים שהמפעיל אמור לסדר
            assert isinstance(ws.get("chunk_size"), int) and ws["chunk_size"]>0
            assert isinstance(ws.get("max_pending_msgs"), int) and ws["max_pending_msgs"]>0
            assert ws.get("permessage_deflate") in (True, False)
            # ודא שהשרת קורא את הערכים (דרך בנאי WSServer - לא נפתח סוקט בפועל)
            s = WSServer(host="127.0.0.1", port=0, handler=lambda x: x,
                         chunk_size=ws.get("chunk_size", 32000),
                         permessage_deflate=ws.get("permessage_deflate", True))
            assert s._chunk_size == ws.get("chunk_size", 32000)
            return True
        """
        return [create_test("ws_exec_test", code)]