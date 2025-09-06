# server/canary_auto_policy.py
from __future__ import annotations
from pydantic import BaseModel, Field

class AutoCanaryPolicy(BaseModel):
    max_error_rate: float = Field(0.02, ge=0.0, le=1.0)  # 2%
    max_p95_ms: int = Field(800, ge=1)
    min_ready_ratio: float = Field(0.90, ge=0.0, le=1.0) # לפחות 90% פודים ready
    step_percent: int = Field(10, ge=1, le=100)
    max_steps: int = Field(10, ge=1)
    hold_seconds: int = Field(30, ge=1)
    consecutive_ok_for_promote: int = Field(2, ge=1)
    rollback_on_first_violation: bool = True