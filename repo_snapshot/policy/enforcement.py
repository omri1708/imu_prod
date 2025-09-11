# policy/enforcement.py
import time, hashlib
from dataclasses import dataclass
from typing import Optional

TRUST_LEVELS = ("untrusted","low","medium","high","system")

@dataclass(frozen=True)
class UserPolicy:
    user_id: str
    trust: str = "low"              # אחת מ: TRUST_LEVELS
    ttl_seconds_t1: int = 7*24*3600  # זיכרון קצר-טווח
    ttl_seconds_t2: int = 90*24*3600 # זיכרון ארוך-טווח
    evidence_required: bool = True   # אין תשובה בלי ראיות
    min_evidence_trust: float = 0.6  # רמת אמון מינימלית בראיה
    max_p95_ms: int = 1500           # SLO
    topic_rate_limit: int = 20       # אירועים/דקה לנושא
    topic_burst: int = 10            # N*burst חוסם פיצוצים
    allow_external_exec: bool = False

def user_hash(user_id:str)->str: return hashlib.sha256(user_id.encode()).hexdigest()[:12]

class Policy:
    def __init__(self): self._by_user = {}
    def upsert(self, p:UserPolicy): self._by_user[p.user_id] = p
    def get(self, user_id:str)->UserPolicy:
        return self._by_user.get(user_id, UserPolicy(user_id=user_id))
    def require(self, user_id:str, *, need_external:bool=False, need_evidence:bool=True):
        p = self.get(user_id)
        if need_external and not p.allow_external_exec:
            from engine.errors import PolicyDenied
            raise PolicyDenied("external_exec_not_allowed")
        if need_evidence and not p.evidence_required:
            # מותר לקשיח – לא נשתמש כאן, כי אצלך דרשת Evidences חובה
            pass
        return p
POLICY = Policy()