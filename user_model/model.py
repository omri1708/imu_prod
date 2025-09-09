# user_model/model.py
from __future__ import annotations
import os, json, time, base64, hashlib, threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# -*- coding: utf-8 -*-

ROOT = "./assurance_store_users"
os.makedirs(ROOT, exist_ok=True)
_lock = threading.RLock()

def _path(uid: str) -> str:
    return os.path.join(ROOT, f"{uid}.json")


def now() -> float: return time.time()

def _sha256(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

class _Crypto:
    """
    Optional Fernet; fallback to XOR (dev-only). For prod, install 'cryptography'.
    """
    def __init__(self, key: Optional[bytes] = None):
        self.mode = "xor"
        self.key = key or b"dev-key-please-replace"
        try:
            from cryptography.fernet import Fernet
            self.mode="fernet"
            self.f = Fernet(base64.urlsafe_b64encode(hashlib.sha256(self.key).digest()))
        except Exception:
            pass
    def enc(self, b: bytes) -> bytes:
        if self.mode=="fernet": return self.f.encrypt(b)
        return bytes([x ^ 0x5A for x in b])
    def dec(self, b: bytes) -> bytes:
        if self.mode=="fernet": return self.f.decrypt(b)
        return bytes([x ^ 0x5A for x in b])

@dataclass
class Consent:
    purpose: str
    granted_ts: float
    ttl_seconds: int
    revoked: bool = False

@dataclass
class Preference:
    key: str
    value: Any
    confidence: float = 0.5
    ts: float = 0.0

class UserStore:
    def __init__(self, root: str = "./assurance_store_users", secret: Optional[bytes] = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.crypto = _Crypto(secret)
    """אחסון פרופיל/פרסונה/מצב משתמש כ-JSON לקובץ אחד פר משתמש."""
 
    def get(self, uid: str) -> Dict[str, Any]:
        p = _path(uid)
        if not os.path.exists(p):
            return {"uid": uid, "profile": {}, "persona": {}, "prefs": {}, "stats": {}}
        try:
            return json.loads(open(p, "r", encoding="utf-8").read())
        except Exception:
            return {"uid": uid, "profile": {}, "persona": {}, "prefs": {}, "stats": {}}

    def set(self, uid: str, data: Dict[str, Any]) -> None:
        with _lock:
            open(_path(uid), "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, indent=2))

    def upsert_persona(self, uid: str, persona_patch: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            st = self.get(uid)
            p = st.get("persona") or {}
            p.update(persona_patch or {})
            st["persona"] = p
            st["persona_ts"] = int(time.time()*1000)
            self.set(uid, st)
            return p

    def append(self, uid: str, record: Dict[str,Any]):
        p = self._path(uid)
        with open(p, "ab") as f:
            enc = self.crypto.enc(json.dumps({"ts": now(), **record}, ensure_ascii=False).encode("utf-8"))
            f.write(enc + b"\n")

    def iter(self, uid: str) -> List[Dict[str,Any]]:
        p = self._path(uid)
        if not p.exists(): return []
        out=[]
        for line in open(p, "rb"):
            out.append(json.loads(self.crypto.dec(line.strip()).decode("utf-8")))
        return out

class UserModel:
    """
    Operational user model: identities, consent records, long-term preferences (with conflict resolution), privacy/TTL.
    """
    def __init__(self, store: UserStore):
        self.store = store

    def identity_register(self, uid: str, traits: Dict[str,Any] | None = None):
        self.store.append(uid, {"evt":"identity", "traits": traits or {}})

    def consent_grant(self, uid: str, purpose: str, ttl_seconds: int):
        self.store.append(uid, {"evt":"consent", "consent": asdict(Consent(purpose, now(), ttl_seconds))})

    def consent_revoke(self, uid: str, purpose: str):
        # append revoked state
        self.store.append(uid, {"evt":"consent", "consent": asdict(Consent(purpose, now(), 0, revoked=True))})

    def has_consent(self, uid: str, purpose: str) -> bool:
        latest=None
        for r in self.store.iter(uid):
            if r.get("evt")=="consent" and r["consent"]["purpose"]==purpose:
                latest = r["consent"]
        if not latest: return False
        if latest.get("revoked"): return False
        if now() - latest["granted_ts"] > latest["ttl_seconds"]: return False
        return True

    def pref_set(self, uid: str, key: str, value: Any, confidence: float = 0.6):
        self.store.append(uid, {"evt":"pref", "pref": asdict(Preference(key, value, confidence, now()))})

    def pref_get(self, uid: str, key: str) -> Optional[Preference]:
        prefs=[r["pref"] for r in self.store.iter(uid) if r.get("evt")=="pref" and r["pref"]["key"]==key]
        if not prefs: return None
        # conflict resolution: pick highest confidence; if tie → most recent
        prefs.sort(key=lambda p: (p["confidence"], p["ts"]))
        p = prefs[-1]
        return Preference(**p)
