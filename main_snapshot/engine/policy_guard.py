# engine/policy_guard.py
# -*- coding: utf-8 -*-
"""
Guard מחבר: לפני כל שלב/קריאה ל-Adapter—נבדוק policy + נעדכן מטריקות p95 + נכריח Grounding כשצריך.
"""
from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
from governance.enforcement import enforce_p95, ratelimit, assert_net, assert_ws, assert_path, hard_ground

def guard_stage(user_id: str, stage: str, started_at: float):
    dt_ms = int((time.time() - started_at) * 1000)
    enforce_p95(user_id, stage, dt_ms)

def guard_claims(user_id: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return hard_ground(user_id, claims)

def guard_net(user_id: str, url: str):
    assert_net(user_id, url)

def guard_ws(user_id: str, host: str):
    assert_ws(user_id, host)

def guard_path(user_id: str, path: str, write: bool = False):
    assert_path(user_id, path, write)

def guard_rate(user_id: str, topic: str, n: int = 1):
    ratelimit(user_id, topic, n)