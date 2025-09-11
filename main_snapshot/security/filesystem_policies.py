# security/filesystem_policies.py
# Deny-by-default מערכת־קבצים, sandbox פר־משתמש + TTL לניקוי.

from __future__ import annotations
import os, time, shutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class PathRule:
    path: str            # בסיס מוחלט
    mode: str            # "ro" | "rw"
    ttl_seconds: int = 0 # TTL לקבצים/תיקיות שנוצרים תחת path

@dataclass
class UserFsPolicy:
    user_id: str
    default_deny: bool = True
    rules: List[PathRule] = field(default_factory=list)
    max_bytes: int = 2 * 1024 * 1024 * 1024

class FsPolicyDB:
    def __init__(self):
        self._by_user: Dict[str, UserFsPolicy] = {}
    def put(self, p: UserFsPolicy):
        self._by_user[p.user_id] = p
    def get(self, user_id: str) -> Optional[UserFsPolicy]:
        return self._by_user.get(user_id)

FS_DB = FsPolicyDB()

def is_path_allowed(user_id: str, path: str, write: bool) -> bool:
    p = FS_DB.get(user_id)
    if not p: return False
    ap = os.path.abspath(os.path.expanduser(path))
    for r in p.rules:
        base = os.path.abspath(os.path.expanduser(r.path))
        if ap.startswith(base):
            if write and r.mode != "rw":
                return False
            return True
    return not p.default_deny

def cleanup_ttl(user_id: str):
    p = FS_DB.get(user_id)
    if not p: return
    now = time.time()
    for r in p.rules:
        base = os.path.abspath(os.path.expanduser(r.path))
        if r.ttl_seconds > 0 and os.path.isdir(base):
            for name in os.listdir(base):
                fp = os.path.join(base, name)
                try:
                    st = os.stat(fp)
                    if now - st.st_mtime > r.ttl_seconds:
                        if os.path.isdir(fp): shutil.rmtree(fp, ignore_errors=True)
                        else: os.remove(fp)
                except FileNotFoundError:
                    pass

# דוגמת sandbox למשתמש דמו:
_default_root = "/mnt/data/imu_repo/var/demo-user"
os.makedirs(_default_root, exist_ok=True)
FS_DB.put(UserFsPolicy(
    user_id="demo-user",
    default_deny=True,
    rules=[PathRule(_default_root, "rw", ttl_seconds=24*3600)],
    max_bytes=512*1024*1024
))