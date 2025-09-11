# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import time

def allowed_scope(scope: str) -> bool:
    return scope in {"t0","t1","t2","t3"}

def apply_ttl(rec: Dict[str,Any]) -> bool:
    ttl = int(rec.get("ttl_s", 0)) or 0
    if ttl <= 0: return True
    return (time.time() - float(rec.get("ts",0))) <= ttl

def scope_filter(rec: Dict[str,Any], allowed: set[str]) -> bool:
    return rec.get("scope") in allowed

def consent_ok(rec: Dict[str,Any]) -> bool:
    return bool(rec.get("consent", True))
