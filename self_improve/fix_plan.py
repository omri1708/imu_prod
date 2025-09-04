# imu_repo/self_improve/fix_plan.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class FixAction:
    path: List[str]          # מסלול בקונפיג (למשל ["ws","chunk_size"])
    op: str                  # "set" | "inc" | "dec"
    value: Any               # ערך יעד/דלתא

@dataclass
class FixPlan:
    reason: str              # "p95_high" | "error_rate_high" | "gate_denied_high" וכו'
    actions: List[FixAction] = field(default_factory=list)
    notes: str = ""
    expected_effect: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self)->Dict[str,Any]:
        return {
            "reason": self.reason,
            "actions": [ {"path":a.path,"op":a.op,"value":a.value} for a in self.actions ],
            "notes": self.notes,
            "expected_effect": self.expected_effect
        }