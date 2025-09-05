# imu_repo/engine/errors.py
from __future__ import annotations


class ResourceRequired(RuntimeError):
    def __init__(self, what: str, how_to_get: str):
        super().__init__(f"resource_required: {what} — {how_to_get}")
        self.what = what
        self.how_to_get = how_to_get
        
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