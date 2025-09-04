# imu_repo/engine/strict_grounded.py
from __future__ import annotations
import hashlib, json
from typing import Any, Callable, Awaitable, Dict, Optional

from engine.evidence_middleware import guarded_handler, current
from user_model.policy import resolve_gate

def _auto_claim(x: Any) -> Dict[str,Any]:
    """
    יוצר ראיה "מקומית־דטרמיניסטית" עבור קלט x.
    - source_url: local://deterministic
    - trust: 1.0 (כי ניתנת לשחזור מלא מתוך כניסת הפונקציה וקוד הדטרמיניזם)
    - ttl_s: קצר (למשל 5s) — נועד כדי לא לאגור ראיות לנצח.
    """
    blob = json.dumps({"input": x}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    return {
        "source_url": "local://deterministic",
        "trust": 1.0,
        "ttl_s": 5,
        "payload": {"derivation": "sha256(input)", "input_hash": h}
    }

async def strict_guarded(handler: Callable[[Any], Awaitable[Any]]|Callable[[Any], Any], *,
                         min_trust: float=0.7) -> Callable[[Any], Awaitable[Dict[str,Any]]]:
    safe = await guarded_handler(handler, min_trust=min_trust)
    async def _wrapped(x: Any) -> Dict[str,Any]:
        cur = current()
        cur.add_evidence("auto_local", _auto_claim(x))
        return await safe(x)
    return _wrapped


async def strict_guarded_for_user(handler: Callable[[Any], Awaitable[Any]]|Callable[[Any], Any],
                                  *,
                                  user_id: Optional[str]) -> Callable[[Any], Awaitable[Dict[str,Any]]]:
    """
    עטיפה דיפולטיבית לכל ה-Pipeline: פר-משתמש.
    מיישמת Strict Grounded עם ספי Evidence ומדיניות max_age_s לפי user_model.policy.
    """
    gate = resolve_gate(user_id)
    min_trust = float(gate["min_trust"])
    max_age_s = int(gate["max_age_s"])
    safe = await guarded_handler(handler, min_trust=min_trust, override_max_age_s=max_age_s)
    async def _wrapped(x: Any) -> Dict[str,Any]:
        cur = current()
        cur.add_evidence("auto_local", _auto_claim(x))
        return await safe(x)
    return _wrapped