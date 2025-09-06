# engine/policy.py
# -*- coding: utf-8 -*-
import time
from typing import List, Dict, Any, Optional
from provenance.store import CASStore, EvidenceMeta
from dataclasses import dataclass

@dataclass
class UserSubspace:
    user_id: str
    trust_level: int  # 0..3
    ttl_seconds: int  # default TTL for artifacts
    allow_exec: bool  # allow actually running installers/commands
    allow_network: bool
    strict_provenance: bool


@dataclass
class RequestContext:
    user: UserSubspace
    reason: str
    now: float = time.time()


class PolicyError(Exception): ...

class AskAndProceedPolicy:
    """
    שם-קוד למדיניות:
    - אם חסר משאב/יכולת: מבקשים הרשאה (ברמת user subspace).
    - אם מותר: מבצעים, רושמים Provenance, וממשיכים.
    - אם אסור/נכשל: חוזרים עם fallback+evidence+reject אבל ממשיכים את ה-pipeline (progression).
    """
    def __init__(self, registry: Dict[str, Dict[str, Any]]):
        self.registry = registry  # name -> {"installer": [...], "min_trust": int, "needs_network": bool}

    def authorize_install(self, ctx: RequestContext, capability: str) -> bool:
        ent = self.registry.get(capability)
        if not ent:
            raise PolicyError(f"unknown_capability:{capability}")
        if ctx.user.trust_level < ent.get("min_trust", 1):
            return False
        if ent.get("needs_network", False) and not ctx.user.allow_network:
            return False
        if not ctx.user.allow_exec:
            return False
        return True

    def ttl_for_capability(self, ctx: RequestContext, capability: str) -> int:
        return min(self.registry.get(capability, {}).get("ttl_hint", 3600), ctx.user.ttl_seconds)

    def validate_adapter_exec(self, ctx: RequestContext, adapter: str, commands: list[str]) -> None:
        # דוגמה: אסור מחיקות מסוכנות, אסור curl|sh ללא hash אלא אם strict_provenance=False וכו'
        dangerous = any(("rm -rf /" in c or "curl | sh" in c) for c in commands)
        if dangerous and ctx.user.strict_provenance:
            raise PolicyError(f"dangerous_command_blocked:{adapter}")

class GroundingPolicy:
    """
    מחייב לפחות ראיה אחת מאומתת (verify_meta==True) ולערך trust>=threshold.
    """
    def __init__(self, trust_threshold: float = 0.6):
        self.trust_threshold = float(trust_threshold)
        self.cas = CASStore()

    def check(self, claims: List[Dict[str, Any]]) -> bool:
        if not claims:
            return False
        ok = False
        for c in claims:
            digest = c.get("sha256")
            if not digest:
                continue
            meta = self.cas.get(digest)
            if not meta:
                continue
            if meta.trust >= self.trust_threshold and self.cas.verify_meta(meta):
                ok = True
            else:
                return False
        return ok