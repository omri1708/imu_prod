# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, glob
from typing import Dict, Any, Iterable, List

def iter_audit_events(audit_dirs: Iterable[str]) -> Iterable[Dict[str,Any]]:
    """
    קורא את audit.jsonl מכל חנויות הקרנל (assurance_store*/audit/audit.jsonl) ומחזיר אירועים.
    פורמט: כל שורה = {"ts":..., "root":..., "op":..., "version":..., "manifest_digest":..., "events":[...]}
    """
    for d in audit_dirs:
        p = os.path.join(d, "audit", "audit.jsonl")
        if not os.path.exists(p): continue
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def summarize(events: Iterable[Dict[str,Any]]) -> Dict[str,Any]:
    summary = {
        "total_commits": 0,
        "resource_required": {},  # what -> count
        "validation_failed": 0,
        "refused_not_grounded": 0,
    }
    for rec in events:
        summary["total_commits"] += 1
        for ev in rec.get("events", []):
            e = ev.get("event")
            if e == "build": pass  # useful for timing if נשמור timing
            if e == "validate":
                # אם היו ולידטורים שנפלו, הקרנל כבר זרק Exception ולא נרשם commit
                pass
        # סריקת הודעות בשדה events אינה מכסה ResourceRequired/Refused שנעצרו לפני commit; לכן נחבר גם לוגים אחרים אם יש.
    return summary
