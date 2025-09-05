# identity/profile_store.py
# -*- coding: utf-8 -*-
import os, json, time, base64, hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from adapters.contracts import ResourceRequired

def _get_cipher(key_path: str):
    try:
        from cryptography.fernet import Fernet
    except Exception:
        raise ResourceRequired("Python 'cryptography' package",
                               "pip install cryptography")
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "wb") as f: f.write(key)
    else:
        with open(key_path, "rb") as f: key = f.read()
    return Fernet(key)

@dataclass
class Consent:
    accepted: bool
    ts: float
    scope: Dict[str, bool]  # {"analytics": True, "ads": False, ...}

@dataclass
class Profile:
    user_id: str
    traits: Dict[str, Any]
    goals: Dict[str, float]
    culture: Dict[str, Any]
    affect: Dict[str, float]
    ttl_sec: int = 90*24*3600  # 90 ימים ברירת מחדל
    created_ts: float = time.time()
    updated_ts: float = time.time()
    consent: Optional[Consent] = None

class ProfileStore:
    def __init__(self, root_dir: str = ".imu/identity", key_file: str = ".imu/keys/enc.key"):
        self.root_dir = root_dir
        self.cipher = _get_cipher(key_file)

    def _path(self, user_id: str) -> str:
        h = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return os.path.join(self.root_dir, f"{h}.bin")

    def save(self, p: Profile):
        os.makedirs(self.root_dir, exist_ok=True)
        blob = json.dumps(asdict(p), ensure_ascii=False).encode("utf-8")
        enc = self.cipher.encrypt(blob)
        with open(self._path(p.user_id), "wb") as f:
            f.write(enc)

    def load(self, user_id: str) -> Optional[Profile]:
        path = self._path(user_id)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            enc = f.read()
        try:
            blob = self.cipher.decrypt(enc)
        except Exception:
            return None
        d = json.loads(blob.decode("utf-8"))
        # TTL
        if d.get("ttl_sec") and (time.time() - d.get("updated_ts", 0)) > d["ttl_sec"]:
            # פג - מוחקים
            try: os.remove(path)
            except Exception: pass
            return None
        # שיחזור dataclasses
        c = d.get("consent")
        consent = None
        if c is not None:
            consent = Consent(**c)
        d["consent"] = consent
        return Profile(**d)

    def set_consent(self, user_id: str, accepted: bool, scope: Dict[str, bool]):
        p = self.load(user_id) or Profile(user_id=user_id, traits={}, goals={}, culture={}, affect={})
        p.consent = Consent(accepted=accepted, ts=time.time(), scope=scope)
        p.updated_ts = time.time()
        self.save(p)

    def update(self, user_id: str, **fields):
        p = self.load(user_id) or Profile(user_id=user_id, traits={}, goals={}, culture={}, affect={})
        for k, v in fields.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.updated_ts = time.time()
        self.save(p)