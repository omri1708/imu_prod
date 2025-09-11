# imu_repo/engine/alerts.py
from __future__ import annotations
from typing import Any, Dict
from engine.audit_log import record_event

def alert(level: str, message: str, meta: Dict[str,Any]) -> None:
    """
    מדווח לאודיט; ניתן להחליף בקלות לשולח מייל/וובהוק.
    """
    record_event("ALERT", {"level": level, "message": message, **(meta or {})}, severity=level.lower())