# imu_repo/engine/evidence_middleware.py
from __future__ import annotations
from typing import Dict, Any, Callable, Awaitable, Optional, List
import os, json, hashlib, hmac, time, secrets

from engine.config import load_config, save_config

from grounded.claims import current
from grounded.gate import GateDenied, enforce_all
from alerts.notifier import alert, metrics_log
from grounded.provenance import sign_evidence, persist_record, verify_signature

LOGS = os.getenv("IMU_LOG_DIR", ".imu_state/logs")
os.makedirs(LOGS, exist_ok=True)

try:
    # ציפייה: יש מודול grounded.claims עם current().add_evidence(...)
    from grounded.claims import current  # type: ignore
except Exception:
    # Fallback קטן — מחסן גלובלי פר-תהליך (לבדיקות)
    import threading
    _local = threading.local()
    def current():
        if not hasattr(_local, "ev"):
            _local.ev = _Claims()
        return _local.ev
    class _Claims:
        def __init__(self): self.buf=[]
        def add_evidence(self, key: str, ev: Dict[str,Any]): self.buf.append((key, ev))
        def drain(self)->List[Dict[str,Any]]:
            out=[]
            for k,e in self.buf:
                out.append(dict(e, key=k))
            self.buf.clear()
            return out

def _ensure_hmac_key(cfg: Dict[str,Any]) -> bytes:
    ev = cfg.setdefault("evidence", {})
    key_hex = ev.get("hmac_key")
    if not key_hex:
        key_hex = secrets.token_hex(32)
        ev["hmac_key"] = key_hex
        save_config(cfg)
    return bytes.fromhex(key_hex)

async def guarded_handler(fn: Callable[[Any], Awaitable[Any]],
                        handler: Callable[[Any], Awaitable[Any]], * ,
                        override_max_age_s: Optional[int] = None,
                        min_trust: float) -> Callable[[Any], Awaitable[Dict[str,Any]]]:

    """
    עוטף handler אסינכרוני ומחייב Evidence Gate לפני החזרת תשובה.
    מחזיר עטיפה שמחייבת Evidences איכותיות/טריות *לפני* RESPOND.
    - עם ראיות: מחתים ושומר Provenance; מחזיר {"text":..., "claims":[...]}.
    תומך ב-override של max_age_s (לשימוש במדיניות פר-משתמש).
    """
    cfg = load_config()
    ev_cfg = cfg.get("evidence", {}) or {}
    required = bool(ev_cfg.get("required", True))
    max_age_s_global = int(cfg.get("guard", {}).get("max_age_s", 3600))
    max_age_s = int(override_max_age_s if override_max_age_s is not None else max_age_s_global)
    secret = _ensure_hmac_key(cfg)

    async def _wrapped(x: Any) -> Dict[str,Any]:
        # אסוף ראיות מה־context
        cur = current()
        try:
            drain = cur.drain()
        except Exception:
            drain = []

        good: List[Dict[str,Any]] = []
        for ev in drain:
            trust = float(ev.get("trust", 0.0))
            age = int(ev.get("ttl_s", 0))  # TTL כאן מפורש "גיל אפקטיבי" שנמדד מול max_age_s
            if trust >= min_trust and age <= max_age_s:
                good.append(ev)

        if required and not good:
            raise PermissionError("evidence_required")

        # הפעל את ה-handler בפועל
        text = await handler(x)

        # חתימה + שמירה
        claims: List[Dict[str,Any]] = []
        for ev in good:
            rec = sign_evidence(secret, {
                "source_url": ev.get("source_url"),
                "trust": float(ev.get("trust", 0.0)),
                "ttl_s": int(ev.get("ttl_s", 0)),
                "key": ev.get("key"),
                "payload": ev.get("payload")
            })
            persist_record(rec)
            # ודא חתימה תקפה
            assert verify_signature(secret, rec) is True
            claims.append({"sha256": rec["sha256"], "sig": rec["sig_hmac_sha256"], "source_url": ev.get("source_url")})

        return {"text": text, "claims": claims}

   
    async def _inner(inp: Any) -> Dict[str,Any]:
        t0 = time.time()
        try:
            # איפוס/איסוף קונטקסט לפי בקשה (פשוט לאיפוס ישיר)
            cur = current()
            cur.clear()
            # הציפייה היא שה-handler עצמו יקרא current().add_evidence(...) עבור כל קביעה מגובה.
            txt = await fn(inp)
            out = enforce_all(txt, min_trust = min_trust)
            dt = (time.time()-t0)*1000.0
            metrics_log("guarded_handler", {"ok": True, "latency_ms": dt, "claims": len(out["claims"])})
            return out
        except GateDenied as e:
            dt = (time.time()-t0)*1000.0
            alert("evidence_gate_denied", severity="high", meta={"reasons": e.reasons, "latency_ms": dt})
            raise
        except Exception as e:
            dt = (time.time()-t0)*1000.0
            alert("handler_failure", severity="high", meta={"error": str(e), "latency_ms": dt})
            raise
    
    return _inner, _wrapped