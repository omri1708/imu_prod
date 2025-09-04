# imu_repo/core/contracts/verifier.py
from __future__ import annotations
from typing import Dict, Any

class ContractViolation(Exception):
    def __init__(self, kind:str, detail:Any=None):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail

class Contracts:
    """Verify that execution obeys declared contracts (resources, correctness, policy)."""

    def __init__(self):
        # ניתן להרחיב עם חוזים נוספים (policy/semantic וכו')
        self.active = ["resources"]

    def check_resources(self, metrics:Dict[str,int], limits:Dict[str,int]) -> None:
        """Validate that metrics are within specified limits."""
        for k,lim in limits.items():
            if metrics.get(k,0) > lim:
                raise ContractViolation("resource_exceeded", {k:metrics.get(k)})
