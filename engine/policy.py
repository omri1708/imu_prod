# engine/policy.py
# -*- coding: utf-8 -*-
import time
from typing import List, Dict, Any, Optional
from provenance.store import CASStore, EvidenceMeta
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import shutil
import os
import subprocess


class EvidenceRef(BaseModel):
    ref: str
    trust: float

class CapabilityResult(BaseModel):
    capability: str
    status: str  # OK | REQUIRED | FAILED
    missing: Optional[List[str]] = None
    hint: Optional[str] = None
    evidence: List[EvidenceRef] = Field(default_factory=list)

class CapabilityRequest(BaseModel):
    capability: str
    user_id: str = "default"
    auto_install: bool = True

# כללי מדיניות קשיחים/רכים
class HardRule(BaseModel):
    name: str
    def enforce(self, adapter: str, cmd: List[str], args: Dict[str, Any]):
        # דוגמה: אסור להריץ פקודה שכוללת rm -rf /
        joined = " ".join(cmd)
        if "rm -rf /" in joined:
            raise RuntimeError(f"policy.hard.{self.name}: blocked dangerous cmd")

class UserSpacePolicy(BaseModel):
    user_id: str
    ttl_seconds: int = 3600
    trust_threshold: float = 0.6
    hard_rules: List[HardRule] = Field(default_factory=lambda: [HardRule(name="no_rm_root")])
    p95_ms: int = 10_000

def evaluate_policy_for_user(user_id: str) -> UserSpacePolicy:
    # ניתן להחמיר לפי משתמש/דומיין
    return UserSpacePolicy(user_id=user_id)

def _need(binary: str) -> Optional[str]:
    return None if shutil.which(binary) else binary

def _sudo_available() -> bool:
    return shutil.which("sudo") is not None

def _try_install(binary: str) -> bool:
    # ניסיון "best effort" לפי פלטפורמה. לא מבטיח.
    # Linux: apt / yum; Mac: brew; Win: winget
    try:
        if shutil.which("apt"):
            return subprocess.call(["sudo","apt","update"]) == 0 and \
                   subprocess.call(["sudo","apt","install","-y", binary]) == 0
        if shutil.which("brew"):
            return subprocess.call(["brew","install", binary]) == 0
        if shutil.which("winget"):
            # ננסה בשם החבילה; למזהים מורכבים מומלץ mapping
            return subprocess.call(["winget","install","-e","--id", binary]) == 0 or \
                   subprocess.call(["winget","install", binary]) == 0
    except Exception:
        return False
    return False

def request_and_continue(req: CapabilityRequest) -> CapabilityResult:
    cap = req.capability.lower()

    # מיפוי דרישות מינימליות לכל יכולת
    REQS = {
        "android": ["java", "javac", "gradle"],
        "ios": ["xcodebuild"],
        "unity": ["unity"],
        "cuda": ["nvidia-smi"],
        "k8s": ["kubectl"],
    }
    missing = []
    for need in REQS.get(cap, []):
        b = _need(need)
        if b: missing.append(b)

    if missing and req.auto_install:
        # ננסה להתקין — "לבקש ולהמשיך": אם אין הרשאות/זמינות, נחזור REQUIRED אבל לא נתקע
        actually_missing = []
        for m in missing:
            ok = _try_install(m)
            if not ok:
                actually_missing.append(m)
        missing = actually_missing

    if missing:
        return CapabilityResult(
            capability=cap, status="REQUIRED", missing=missing,
            hint="Install the missing SDKs/tools or provide a container with them."
        )

    return CapabilityResult(capability=cap, status="OK", evidence=[EvidenceRef(ref=f"bin:{cap}", trust=0.7)])


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