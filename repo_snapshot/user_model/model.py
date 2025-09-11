# user_model/model.py
from __future__ import annotations
import os
import json
import time
import base64
import hashlib
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

# -*- coding: utf-8 -*-

"""
User model storage with a clear separation between:
  • Profile (single pretty-printed JSON per user):   <root>/<uid>.json
  • Event log (encrypted JSONL, one record per line): <root>/<uid>.events

- Thread-safe writes via a module-level RLock
- Fernet (AEAD) encryption for events
- Optional legacy XOR *read-only* fallback behind an env flag
- Iteration yields decrypted JSON dicts
- Profile writes are atomic (tmp + replace)
"""

ROOT_DEFAULT = "./assurance_store_users"
_lock = threading.RLock()

ENV_FERNET_KEY = "USERSTORE_FERNET_KEY"          # base64 urlsafe 32-byte key
ENV_ALLOW_XOR_READ = "USERSTORE_ALLOW_XOR_READ"  # "1" to enable legacy XOR read
KEYFILE_NAME = ".fernet.key"                     # stored under root/

def now() -> float:
    return time.time()

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)

class _Crypto:
    """
    Fernet AEAD for encryption/decryption.
    - Key precedence: explicit 'key' arg > ENV USERSTORE_FERNET_KEY > keyfile under root > generate+persist
    - Legacy XOR *read only* fallback controlled by ENV USERSTORE_ALLOW_XOR_READ="1" (for old logs).
    """

    def __init__(self, root: Path, key: Optional[bytes] = None):
        self.root = root
        self.allow_xor_read = os.getenv(ENV_ALLOW_XOR_READ, "0") == "1"

        # Resolve key (Fernet key must be urlsafe base64 32 bytes)
        k = key or os.getenv(ENV_FERNET_KEY, "").encode("utf-8") or self._load_or_create_keyfile()
        try:
            from cryptography.fernet import Fernet  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "cryptography is required for Fernet. Install with: pip install cryptography"
            ) from e

        # Accept either a proper Fernet key or derive from arbitrary secret by scrypt->32B->b64
        if not self._looks_like_fernet_key(k):
            k = self._derive_fernet_key_from_secret(k)

        self._fernet_key = k
        self.f = Fernet(k)

    def _keyfile_path(self) -> Path:
        return self.root / KEYFILE_NAME

    def _load_or_create_keyfile(self) -> bytes:
        kp = self._keyfile_path()
        if kp.exists():
            raw = kp.read_bytes().strip()
            return raw
        # create new Fernet key and persist with 0600
        from cryptography.fernet import Fernet  # type: ignore
        k = Fernet.generate_key()
        # Ensure directory exists (root already created by UserStore)
        with open(kp, "wb") as fh:
            fh.write(k)
        try:
            os.chmod(kp, 0o600)
        except Exception:
            pass
        return k

    @staticmethod
    def _looks_like_fernet_key(k: bytes) -> bool:
        # Heuristic: Fernet key is urlsafe base64 and decodes to 32 bytes
        try:
            dd = base64.urlsafe_b64decode(k)
            return len(dd) == 32
        except Exception:
            return False

    @staticmethod
    def _derive_fernet_key_from_secret(secret: bytes) -> bytes:
        """Derive a Fernet key from arbitrary secret via scrypt."""
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt  # type: ignore
        from cryptography.hazmat.backends import default_backend  # type: ignore

        # Fixed salt path: per-installation salt under the hood of the derived key
        # (for stronger security, keep a separate salt file; here we bake a constant tag)
        salt = hashlib.sha256(b"userstore-derivation-salt").digest()
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1, backend=default_backend())
        key32 = kdf.derive(secret)
        return base64.urlsafe_b64encode(key32)

    # ---------- API ----------
    def enc(self, b: bytes) -> bytes:
        return self.f.encrypt(b)

    def dec(self, b: bytes) -> bytes:
        # Try Fernet first
        try:
            return self.f.decrypt(b)
        except Exception:
            if self.allow_xor_read:
                # Legacy XOR fallback: previous implementation used constant ^0x5A per-byte.
                return bytes([x ^ 0x5A for x in b])
            raise

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
    """Profile + Encrypted events storage."""

    def __init__(self, root: str = ROOT_DEFAULT, secret: Optional[bytes] = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.crypto = _Crypto(self.root, secret)

    # ---------- paths ----------
    def _profile_path(self, uid: str) -> Path:
        return self.root / f"{uid}.json"

    def _events_path(self, uid: str) -> Path:
        return self.root / f"{uid}.events"

    # ---------- profile (single JSON) ----------
    def get(self, uid: str) -> Dict[str, Any]:
        p = self._profile_path(uid)
        if not p.exists():
            return {"uid": uid, "profile": {}, "persona": {}, "prefs": {}, "stats": {}}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt or partial file → return empty structure (caller may choose to overwrite)
            return {"uid": uid, "profile": {}, "persona": {}, "prefs": {}, "stats": {}}

    def set(self, uid: str, data: Dict[str, Any]) -> None:
        with _lock:
            _atomic_write_text(
                self._profile_path(uid),
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def upsert_persona(self, uid: str, persona_patch: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            st = self.get(uid)
            p = st.get("persona") or {}
            p.update(persona_patch or {})
            st["persona"] = p
            st["persona_ts"] = int(time.time() * 1000)
            self.set(uid, st)
            return p

    # ---------- events (encrypted JSONL) ----------
    def append(self, uid: str, record: Dict[str, Any]) -> None:
        ep = self._events_path(uid)
        payload = json.dumps({"ts": now(), **record}, ensure_ascii=False).encode("utf-8")
        enc = self.crypto.enc(payload)
        with _lock, ep.open("ab") as f:
            f.write(enc + b"\n")

    def iter(self, uid: str) -> List[Dict[str, Any]]:
        ep = self._events_path(uid)
        if not ep.exists():
            return []
        out: List[Dict[str, Any]] = []
        with ep.open("rb") as fh:
            for line in fh:
                line = line.rstrip(b"\n")
                if not line:
                    continue
                try:
                    out.append(json.loads(self.crypto.dec(line).decode("utf-8")))
                except Exception:
                    # Skip malformed or undecryptable lines instead of crashing the caller
                    continue
        return out

class UserModel:
    """Operational user model: identities, consent, preferences (with conflict resolution and TTL)."""

    def __init__(self, store: UserStore):
        self.store = store

    # ---- identity ----
    def identity_register(self, uid: str, traits: Dict[str, Any] | None = None) -> None:
        self.store.append(uid, {"evt": "identity", "traits": traits or {}})

    # ---- consent ----
    def consent_grant(self, uid: str, purpose: str, ttl_seconds: int) -> None:
        self.store.append(uid, {"evt": "consent", "consent": asdict(Consent(purpose, now(), ttl_seconds))})

    def consent_revoke(self, uid: str, purpose: str) -> None:
        self.store.append(uid, {"evt": "consent", "consent": asdict(Consent(purpose, now(), 0, revoked=True))})

    def has_consent(self, uid: str, purpose: str) -> bool:
        latest = None
        for r in self.store.iter(uid):
            if r.get("evt") == "consent" and r.get("consent", {}).get("purpose") == purpose:
                latest = r["consent"]
        if not latest:
            return False
        if latest.get("revoked"):
            return False
        if now() - latest["granted_ts"] > latest["ttl_seconds"]:
            return False
        return True

    # ---- preferences ----
    def pref_set(self, uid: str, key: str, value: Any, confidence: float = 0.6) -> None:
        self.store.append(uid, {"evt": "pref", "pref": asdict(Preference(key, value, confidence, now()))})

    def pref_get(self, uid: str, key: str) -> Optional[Preference]:
        prefs = [r["pref"] for r in self.store.iter(uid) if r.get("evt") == "pref" and r["pref"].get("key") == key]
        if not prefs:
            return None
        # Conflict resolution: highest confidence; tie → most recent
        prefs.sort(key=lambda p: (p.get("confidence", 0.0), p.get("ts", 0.0)))
        p = prefs[-1]
        return Preference(**p)
