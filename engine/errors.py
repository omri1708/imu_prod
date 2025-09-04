# imu_repo/engine/errors.py
from __future__ import annotations

class GuardRejection(Exception):
    """נזרק כאשר אסור להחזיר תגובה (חוסר ראיות/חוסר אמון/חוסר טריות/מדיניות)."""
    def __init__(self, reason: str, details: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}

class PolicyError(Exception):
    """שגיאת מדיניות כללית."""
    pass