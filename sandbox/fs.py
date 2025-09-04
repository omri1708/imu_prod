# imu_repo/sandbox/fs.py
from __future__ import annotations
import os, io, errno
from typing import Optional, List
from engine.policy_ctx import get_user

ROOT = "/mnt/data/imu_repo/workspaces"

def _user_root(user_id: str) -> str:
    p = os.path.join(ROOT, user_id)
    os.makedirs(p, exist_ok=True)
    return p

def _norm(user_id: str, rel: str) -> str:
    if not rel:
        raise ValueError("empty_path")
    base = os.path.abspath(_user_root(user_id))
    target = os.path.abspath(os.path.join(base, rel))
    if not target.startswith(base + os.sep) and target != base:
        raise PermissionError("fs_escape_detected")
    return target

def write_text(rel_path: str, text: str, *, user_id: Optional[str] = None, exist_ok_parent: bool = True) -> str:
    uid = user_id or (get_user() or "anon")
    p = _norm(uid, rel_path)
    d = os.path.dirname(p)
    if not os.path.exists(d):
        if exist_ok_parent:
            os.makedirs(d, exist_ok=True)
        else:
            raise FileNotFoundError(errno.ENOENT, "parent_missing", d)
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p

def read_text(rel_path: str, *, user_id: Optional[str] = None) -> str:
    uid = user_id or (get_user() or "anon")
    p = _norm(uid, rel_path)
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()

def list_tree(rel_dir: str = ".", *, user_id: Optional[str] = None) -> List[str]:
    uid = user_id or (get_user() or "anon")
    root = _norm(uid, rel_dir)
    out: List[str] = []
    for base, _dirs, files in os.walk(root):
        for fn in files:
            out.append(os.path.relpath(os.path.join(base, fn), start=_user_root(uid)))
    return sorted(out)