# imu_repo/engine/gates/min_evidence.py
from __future__ import annotations
from typing import Dict, Any, List

class MinEvidenceGate:
    """
    מכריח מספר מינימלי של 'קינדי ראיות' (keys) בתוך evidence.
      config = {"kinds": ["service_tests","perf_summary","ui_accessibility","official_api"], "min": 3}
    מעבר = מתקיים min<= מספר הקינדים שנמצאו בפועל.
    """
    def __init__(self, kinds: List[str], min_required: int):
        self.kinds = list(kinds)
        self.min_required = int(min_required)

    def check(self, evidence: Dict[str,Any]) -> Dict[str,Any]:
        found = 0
        present=[]
        for k in self.kinds:
            if k in evidence and evidence[k]:
                found += 1
                present.append(k)
        ok = (found >= self.min_required)
        return {"ok": ok, "found": found, "present": present, "need": self.min_required}