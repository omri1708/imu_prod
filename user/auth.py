# imu_repo/user/auth.py
from __future__ import annotations
import os, json, time, secrets, hashlib, hmac, base64
from typing import Dict, Any, Optional, List


class AuthError(Exception): ...


def _b64(b: bytes) -> str: return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _ub64(s: str) -> bytes:
    pad = "="*((4 - (len(s)%4))%4); return base64.urlsafe_b64decode((s+pad).encode())

class UserStore:
    """
    רישום משתמשים, הרשאות, ומדיניות הסכמה/פרטיות.
    נשמר ב- .imu_state/users.json
    """
    def __init__(self, path: str = ".imu_state/users.json", secret_path: str = ".imu_state/auth.key"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path,"w",encoding="utf-8") as f: json.dump({"users":{}}, f)
        if not os.path.exists(secret_path):
            with open(secret_path,"wb") as f: f.write(os.urandom(32))
        with open(secret_path,"rb") as f: self._secret = f.read()
        self._load()

    def _load(self):
        with open(self.path,"r",encoding="utf-8") as f: self.db = json.load(f)

    def _save(self):
        with open(self.path,"w",encoding="utf-8") as f: json.dump(self.db, f, ensure_ascii=False, indent=2)

    def ensure_user(self, user_id: str, roles: Optional[List[str]] = None, consent: Optional[Dict[str,Any]] = None):
        u = self.db["users"].get(user_id)
        if not u:
            u = {"roles": roles or ["user"], "consent": consent or {"memory": True, "analytics": True}, "created_at": time.time()}
            self.db["users"][user_id] = u; self._save()
        return u

    def set_consent(self, user_id: str, consent: Dict[str,Any]):
        self.ensure_user(user_id)
        self.db["users"][user_id]["consent"] = consent
        self._save()

    def get(self, user_id: str) -> Optional[Dict[str,Any]]:
        return self.db["users"].get(user_id)

    def has_role(self, user_id: str, role: str) -> bool:
        u=self.get(user_id); 
        return bool(u and role in u.get("roles",[]))

    # ---- JWT-like מינימלי (HMAC) ----
    def issue_token(self, user_id: str, ttl_s: int = 86400) -> str:
        payload = {"sub":user_id,"exp":int(time.time())+ttl_s}
        p=_b64(json.dumps(payload).encode()); h=_b64(hmac.new(self._secret, p.encode(), hashlib.sha256).digest())
        return f"{p}.{h}"

    def verify_token(self, token: str) -> str:
        try:
            p, h = token.split(".")
            exp_sig = hmac.new(self._secret, p.encode(), hashlib.sha256).digest()
            if not hmac.compare_digest(exp_sig, _ub64(h)):
                raise AuthError("bad_sig")
            payload = json.loads(_ub64(p).decode())
            if int(payload["exp"]) < time.time(): raise AuthError("expired")
            user_id = payload["sub"]
            if not self.get(user_id): raise AuthError("no_user")
            return user_id
        except Exception as e:
            raise AuthError(str(e))

class AuthStore:
    """אחסון משתמשים ותפקידים (JSON)."""
    def __init__(self, path: str = ".imu_state/auth/users.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f: json.dump({}, f)

    def _load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f: return json.load(f)
    def _save(self, data: Dict[str, Any]):
        with open(self.path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

    def create_user(self, user_id: str, password: str, roles: list[str] | None = None):
        db = self._load()
        if user_id in db: raise AuthError("user_exists")
        salt = secrets.token_bytes(16)
        pwd = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        db[user_id] = {
            "salt": salt.hex(),
            "pwd": pwd.hex(),
            "roles": roles or ["user"],
            "consent": False,
            "created_ts": time.time(),
            "tokens": {}
        }
        self._save(db)

    def grant_consent(self, user_id: str):
        db = self._load()
        if user_id not in db: raise AuthError("no_such_user")
        db[user_id]["consent"] = True
        self._save(db)

    def revoke_consent(self, user_id: str):
        db = self._load()
        if user_id not in db: raise AuthError("no_such_user")
        db[user_id]["consent"] = False
        self._save(db)

    def authenticate(self, user_id: str, password: str) -> str:
        db = self._load()
        u = db.get(user_id)
        if not u: raise AuthError("no_such_user")
        salt = bytes.fromhex(u["salt"])
        expect = bytes.fromhex(u["pwd"])
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        if not hmac.compare_digest(expect, got): raise AuthError("bad_credentials")
        tok = secrets.token_urlsafe(24)
        u["tokens"][tok] = {"ts": time.time()}
        self._save(db)
        return tok

    def authorize(self, user_id: str, token: str, need_role: str | None = None) -> bool:
        db = self._load()
        u = db.get(user_id)
        if not u: return False
        if token not in u["tokens"]: return False
        if need_role and need_role not in u["roles"]: return False
        return True

    def delete_user(self, user_id: str):
        db = self._load(); db.pop(user_id, None); self._save(db)
