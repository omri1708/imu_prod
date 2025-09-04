# imu_repo/engine/errors.py
from __future__ import annotations

class GuardRejection(Exception):
    def __init__(self, reason: str, details: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}

class BudgetExceeded(Exception):
    def __init__(self, capability: str, required: float, available: float):
        super().__init__(f"budget_exceeded:{capability}:{required}>{available}")
        self.capability = capability
        self.required = float(required)
        self.available = float(available)

class GuardRejection(Exception):
    """נזרק כאשר אסור להחזיר תגובה (חוסר ראיות/חוסר אמון/חוסר טריות/מדיניות)."""
    def __init__(self, reason: str, details: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}

class PolicyError(Exception):
    """שגיאת מדיניות כללית."""
    pass