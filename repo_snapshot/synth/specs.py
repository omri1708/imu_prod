# imu_repo/synth/specs.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Contract:
    name: str
    schema: Optional[Dict[str,Any]] = None
    # חיזוק: אפשר לדרוש אמון מינימלי לראיות מסוימות ברמת החוזה
    evidence_min_trust: Dict[str, float] = field(default_factory=dict)

@dataclass
class BuildSpec:
    name: str
    kind: str
    language_pref: List[str] = field(default_factory=lambda: ["python"])
    ports: List[int] = field(default_factory=list)
    endpoints: Dict[str,str] = field(default_factory=dict)
    contracts: List[Contract] = field(default_factory=list)
    # מפתחות עדות שחובה שיופיעו לפני rollout
    evidence_requirements: List[str] = field(default_factory=list)
    # חדש: איסוף ראיות חיצוני במהלך הפייפליין
    # [{"key":"ext_doc","url":"https://example.com/api/status"} , ...]
    external_evidence: List[Dict[str,str]] = field(default_factory=list)