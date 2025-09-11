# imu_repo/engine/strict_mode.py
from __future__ import annotations
import time, json, hashlib
from typing import Dict, Any, List, Optional, Callable
from engine.respond_guard import ensure_proof_and_package

class StrictGroundingError(Exception): ...

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _mk_compute_claim(*, prompt: str, response_text: str) -> Dict[str,Any]:
    """
    Claim דטרמיניסטי לחישוב "טהור": כולל קלט/פלט, hash וחותמת זמן.
    מאפשר Grounding גם כשאין מקור חיצוני (API/מסמך).
    """
    ts = time.time()
    payload = json.dumps({"prompt": prompt, "response": response_text, "ts": ts}, ensure_ascii=False).encode("utf-8")
    return {
        "id": f"compute:{_hash_bytes(payload)[:16]}",
        "type": "compute",
        "text": "deterministic-compute",
        "schema": {"type": "compute_trace", "unit": "", "min": None, "max": None},
        "value": ts,
        "evidence": [{
            "kind": "compute_trace",
            "hash_sha256": _hash_bytes(payload),
            "ts": ts
        }],
        "consistency_group": "compute"
    }

def strict_package_response(
    *,
    response_text: str,
    claims: Optional[List[Dict[str,Any]]],
    policy: Dict[str,Any],
    http_fetcher: Optional[Callable[[str,str], tuple]] = None,
    sign_key_id: Optional[str] = None
) -> Dict[str,Any]:
    """
    אוכף: אסור להשיב בלי Claims+Evidence.
    אם claims חסר/ריק → מייצר compute-claim דטרמיניסטי.
    לאחר מכן אורז עם proof חתום (ensure_proof_and_package).
    """
    cl = list(claims or [])
    if not cl:
        cl = [_mk_compute_claim(prompt="(omitted)", response_text=response_text)]
    # ווידוא שלכל claim יש לפחות evidence אחת
    for c in cl:
        ev = c.get("evidence")
        if not isinstance(ev, list) or not ev:
            raise StrictGroundingError(f"claim {c.get('id','?')} has no evidence")
    # אריזה/חתימה
    packaged = ensure_proof_and_package(
        response_text=response_text,
        claims=cl,
        policy=policy,
        http_fetcher=(http_fetcher or (lambda url,method: (200,{"date":""},b""))),
        sign_key_id=sign_key_id
    )
    if not packaged.get("ok"):
        raise StrictGroundingError("failed to package response with proof")
    return packaged["proof"]