# server/canary_auto_policy.py
# מדיניות Auto-Canary: ספי שגיאות/latency ופרטי צעדים.
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class AutoCanaryPolicy(BaseModel):
    # thresholds
    max_error_rate: float = Field(0.02, ge=0.0, le=1.0)      # 2%
    max_p95_ms: int = Field(800, ge=1)                       # 800ms
    # step plan
    step_percent: int = Field(10, ge=1, le=100)              # +10% בכל צעד
    max_steps: int = Field(10, ge=1)                         # עד 10 צעדים
    hold_seconds: int = Field(30, ge=1)                      # המתנה בין צעדים
    # promote/rollback
    consecutive_ok_for_promote: int = Field(2, ge=1)         # כמה מחזורים OK לפני promote
    rollback_on_first_violation: bool = True