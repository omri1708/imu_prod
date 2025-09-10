# user_model/consciousness.py

# imu_repo/user_model/consciousness.py
from __future__ import annotations
from typing import Dict, Any
from user_model.memory_store import get_profile

def merge_beliefs(uid: str) -> Dict[str,Any]:
    """תמונת 'מודעות' מינימלית: העדפות (T1) + אמונות/מטרות (T2) כרמות בין 0..1."""
    prof = get_profile(uid)
    mood = prof["beliefs"].get("mood", 0.5)  # מצב רגש כללי (0..1)
    culture = prof["beliefs"].get("culture_context", 0.5)
    goals = {
        "latency_sensitive": prof["beliefs"].get("latency_sensitive", 0.5),
        "accuracy_strict":   prof["beliefs"].get("accuracy_strict", 0.5)
    }
    return {
        "prefs": prof["pref"],
        "beliefs": prof["beliefs"],
        "mood": mood,
        "culture": culture,
        "goals": goals
    }

def route_decision(uid: str, *, base_p95_ms: float) -> Dict[str,Any]:
    """
    דוגמה: התאמת יעד p95 לפי מטרות/רגש. "strict" → p95 נמוך יותר.
    """
    state = merge_beliefs(uid)
    strict = max(state["goals"]["accuracy_strict"], state["beliefs"].get("quality_strict", 0.0))
    # הפחתה עד 30% יעד p95
    factor = 1.0 - 0.3*strict
    target = max(200.0, base_p95_ms*factor)
    return {"p95_target_ms": target, "factor": factor, "state": state}