# imu_repo/self_improve/executors/base.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from self_improve.fix_plan import FixPlan, FixAction

class Executor:
    domain: str = "base"
    def can_handle(self, action: FixAction) -> bool:
        return False
    def apply_actions(self, actions: List[FixAction]) -> Dict[str,Any]:
        """
        מחזיר {"changed": bool, "details": {...}}
        """
        raise NotImplementedError
    def generate_tests(self, actions: List[FixAction]) -> List[Tuple[str,str]]:
        """
        ייצור בדיקות: [(module_path, module_name)]
        """
        return []