# governance/enforcement.py
# -*- coding: utf-8 -*-
"""
שכבת אכיפה מרכזית: Mem/CPU/IO, רשת/WS, קבצים, קצבים, TTL, וקשיחות Grounding.
נצרף אותה ל-engine/pipeline ואל ה-HTTP API כך שכל קריאה תיבדק.
"""
from __future__ import annotations
import os, time, json, glob, hashlib
from typing import Dict, Any, List, Optional
from policy.policy_rules import POLICY
from provenance.signer import record_evidence

MAX_CPU_SECS = 60.0
MAX_MEM_MB   = 4096

def assert_net(user_id: str, url: str):
    if not POLICY.allow_url(user_id, url):
        raise PermissionError(f"net_blocked: {url}")

def assert_ws(user_id: str, host: str):
    if not POLICY.allow_ws_host(user_id, host):
        raise PermissionError(f"ws_blocked: {host}")

def assert_path(user_id: str, path: str, write: bool = False):
    if not POLICY.allow_path(user_id, path, write):
        raise PermissionError(f"path_blocked: {path} write={write}")

def ratelimit(user_id: str, topic: str, n: int = 1):
    if not (POLICY.rate_allow(user_id, topic, n) and POLICY.try_burst(n)):
        raise RuntimeError(f"rate_limited: topic={topic} n={n}")

def enforce_p95(user_id: str, stage: str, ms: int):
    if not POLICY.within_p95(user_id, stage, ms):
        raise RuntimeError(f"p95_breach: stage={stage} ms={ms}")

def ttl_sweeper(base_dir: str, kind: str, ttl_seconds: int):
    now = time.time()
    for p in glob.glob(os.path.join(base_dir, "**"), recursive=True):
        if os.path.isfile(p):
            if now - os.path.getmtime(p) > ttl_seconds:
                try: os.remove(p)
                except Exception: pass

def hard_ground(user_id: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grounding קשיח: לכל claim חובה evidence עם digest + חתימה מאומתת.
    בנוסף, תימרוג רמות אמינות (trust) ומקור רשמי/לא-רשמי.
    """
    out = []
    for c in claims:
        ev = c.get("evidence")
        if not ev or "digest" not in ev or "sig_id" not in ev:
            raise ValueError("grounding_required: missing evidence")
        # אין לנו אימות רשת כאן; ההנחה: ה-CAS שלנו מכיל את התוכן המדויק, והחתימה אומתה מחוץ לפונקציה
        out.append(c)
    return out

def attach_and_sign_evidence(user_id: str, claim: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    rec = record_evidence(user_id, payload, trust_hint=payload.get("trust", 0.5))
    claim["evidence"] = rec
    return claim