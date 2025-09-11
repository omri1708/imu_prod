# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

@dataclass
class SessionState:
    stage: str = "idle"
    slots: Dict[str,Any] = field(default_factory=dict)
    pending_q: Optional[str] = None

    def need(self, key:str, question:str) -> bool:
        if key in self.slots and self.slots[key] not in (None,""): return False
        self.pending_q = question
        return True

    def to_json(self) -> Dict[str,Any]:
        return asdict(self)
