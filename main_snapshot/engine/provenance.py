# imu_repo/engine/provenance.py
from __future__ import annotations
import time, hmac, hashlib
from typing import Any, Dict, Optional, Tuple, List
import hashlib, json, time, os
from dataclasses import dataclass

from pydantic import BaseModel, Field


class ProvenanceError(Exception): ...
class SignatureError(Exception): ...

# רמות הוכחה (גבוה יותר = חזק יותר)
# L0: inline בלבד; L1: HTTP עם מטא תקין; L2: חתימה תקפה; L3: חתימה+חלון-זמן/anti-replay
L0_INLINE = 0
L1_HTTP_META = 1
L2_SIGNED = 2
L3_SIGNED_FRESH = 3

@dataclass
class Evidence(BaseModel):
    kind: str        # "installer_log" | "artifact" | "command_plan" | "signature"
    content: bytes
    meta: Dict[str, Any]
    claim: str
    source: str
    trust: float
    ttl_seconds: int = 3600
    timestamp: float = Field(default_factory=lambda: time.time())
    extra: Dict[str, Any] = Field(default_factory=dict)
    signature: Optional[str] = None
    cas_key: Optional[str] = None

class ProvenanceStore:
    def __init__(self, root="var/prov"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._store: Dict[str, Evidence] = {}

    def _cas(self, ev: Evidence) -> str:
        blob = json.dumps({"claim":ev.claim,"source":ev.source,"extra":ev.extra}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def add_evidence(self, ev: Evidence) -> str:
        ev.cas_key = self._cas(ev)
        # חתימה "פנימית" פשוטה (ללא PKI — ניתן להחליף ל־ed25519)
        ev.signature = hashlib.sha256((ev.cas_key + "|sig").encode()).hexdigest()
        self._store[ev.cas_key] = ev
        return ev.cas_key

    def get(self, cas_key: str) -> Optional[Evidence]:
        return self._store.get(cas_key)

    def list(self) -> List[Evidence]:
        # מסנן פריטים שפג תוקפם
        now = time.time()
        return [e for e in self._store.values() if (e.timestamp + e.ttl_seconds) > now]

    def put(self, ev: Evidence) -> str:
        h = hashlib.sha256(ev.content).hexdigest()
        path = os.path.join(self.root, h[:2], h[2:4])
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"{h}.json")
        doc = {
            "hash": h,
            "kind": ev.kind,
            "meta": ev.meta,
            "ts": time.time(),
        }
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        # נשמור גם את ה-payload כקובץ
        with open(os.path.join(path, f"{h}.bin"), "wb") as f:
            f.write(ev.content)
        return h


def _hmac_ok(message: bytes, hex_sig: str, secret: bytes, algo: str="sha256") -> bool:
    try:
        digestmod = getattr(hashlib, algo)
    except AttributeError:
        raise SignatureError(f"unsupported hash algo: {algo}")
    mac = hmac.new(secret, message, digestmod)
    try:
        calc = mac.hexdigest()
    except Exception:
        calc = mac.digest().hex()
    # השוואה חסינת timing
    return hmac.compare_digest(calc.lower(), (hex_sig or "").lower())


def verify_signature(e: Dict[str,Any], policy: Dict[str,Any]) -> bool:
    """
    אימות חתימה HMAC-SHA256:
    evidence["sig"] = hex; evidence["key_id"]=str; evidence["signed_fields"]=[...]
    policy["signing_keys"] = {"key_id": {"secret_hex":"...", "algo":"sha256"}}
    ההודעה: concatenation של הערכים בשדות signed_fields באותו סדר (ב־utf8).
    """
    key_id = e.get("key_id")
    sig = e.get("sig")
    fields = e.get("signed_fields")
    if not (isinstance(key_id, str) and isinstance(sig, str) and isinstance(fields, list) and fields):
        return False
    keys = policy.get("signing_keys") or {}
    entry = keys.get(key_id)
    if not entry:
        return False
    sec_hex = entry.get("secret_hex")
    algo = entry.get("algo", "sha256")
    if not isinstance(sec_hex, str):
        return False
    try:
        secret = bytes.fromhex(sec_hex)
    except Exception:
        return False
    # message: חיבור הערכים של השדות
    parts = []
    for f in fields:
        v = e
        for seg in str(f).split("."):
            v = v.get(seg) if isinstance(v, dict) else None
        if isinstance(v, (bytes, bytearray)):
            parts.append(bytes(v))
        elif v is None:
            parts.append(b"")
        else:
            parts.append(str(v).encode("utf-8"))
    message = b"\x1f".join(parts)
    return _hmac_ok(message, sig, secret, algo)


def evidence_provenance_level(e: Dict[str,Any], policy: Dict[str,Any], *, now_ts: Optional[float]=None) -> int:
    """
    מדרג רמת מקור:
      - inline ללא חתימה → L0
      - http מטא תקין (סטטוס/גיל) → L1
      - יש חתימה תקפה → L2
      - חתימה + fresh_ts בתוך חלון → L3
    """
    kind = e.get("kind")
    if kind == "inline":
        lvl = L0_INLINE
    elif kind == "http":
        lvl = L1_HTTP_META
        # אם יש אימות חתימה → L2/L3
        if verify_signature(e, policy):
            lvl = L2_SIGNED
            # fresh window (anti replay)
            fresh_s = policy.get("signature_fresh_window_sec")
            if isinstance(fresh_s, (int,float)):
                now = now_ts or time.time()
                ts = None
                # השדה החתום יכול לכלול meta.header.Date או שדה יחודי evidence.ts
                ts = e.get("ts") if isinstance(e.get("ts"), (int,float)) else None
                if ts is None:
                    # ניסיון לחלץ מתאריך מטא אם נחתם
                    hdr = (e.get("headers") or {}).get("date") if isinstance(e.get("headers"), dict) else None
                    # השארנו לפרובנס החיצוני; כאן נדרש שדה ts מפורש אם רוצים L3
                if isinstance(ts, (int,float)) and (now - float(ts) <= float(fresh_s)):
                    lvl = L3_SIGNED_FRESH
    else:
        lvl = L0_INLINE
    return int(lvl)


def enforce_min_provenance(claim: Dict[str,Any], policy: Dict[str,Any], *, now_ts: Optional[float]=None) -> None:
    """
    אוכף ל־claim רמת מינימום (ברירת מחדל כלל־מערכתית או פר claim_type).
    policy:
      - "min_provenance_level": int
      - "min_provenance_by_type": {"latency": 2, ...}
    """
    req = int(policy.get("min_provenance_level", L1_HTTP_META))
    ctyp = claim.get("type")  # אופציונלי
    by = policy.get("min_provenance_by_type") or {}
    if isinstance(ctyp, str) and (ctyp in by):
        req = int(by[ctyp])
    evs = claim.get("evidence") or []
    max_lvl = -1
    for e in evs:
        lvl = evidence_provenance_level(e, policy, now_ts=now_ts)
        if lvl > max_lvl:
            max_lvl = lvl
    if max_lvl < req:
        raise ProvenanceError(f"provenance_fail: need>={req}, got {max_lvl}")