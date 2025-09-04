# imu_repo/adapters/fs_sandbox.py
from __future__ import annotations
import os, io, json, time, errno, shutil
from typing import Optional, Dict, Any



class FSError(Exception): ...
class FSAccessDenied(FSError): ...
class FSPathError(FSError): ...

class FSConfig:
    def __init__(self,
                 root: str = ".imu_state/fsroot",
                 allow_rel: Optional[list[str]] = None,
                 max_file_kb: int = 1024*32,
                 ttl_sec_default: int = 0):
        self.root = root
        self.allow_rel = allow_rel or ["workspace", "scratch", "logs", "cache"]
        self.max_file_kb = int(max_file_kb)
        self.ttl_sec_default = int(ttl_sec_default)

_fs = FSConfig()

def _ensure_root():
    os.makedirs(_fs.root, exist_ok=True)
    for d in _fs.allow_rel:
        os.makedirs(os.path.join(_fs.root, d), exist_ok=True)

def _safe_join(rel_path: str) -> str:
    _ensure_root()
    if rel_path.startswith("/"): raise FSPathError("absolute_path_forbidden")
    p = os.path.normpath(rel_path)
    parts = p.split(os.sep)
    if not parts or parts[0] not in _fs.allow_rel:
        raise FSAccessDenied("top_level_dir_not_allowed")
    full = os.path.normpath(os.path.join(_fs.root, p))
    if not full.startswith(os.path.abspath(_fs.root)):
        raise FSAccessDenied("path_escape")
    return full

def write_text(rel_path: str, text: str, ttl_sec: Optional[int] = None):
    full = _safe_join(rel_path)
    data = text.encode("utf-8")
    if len(data) > _fs.max_file_kb*1024:
        raise FSError("file_too_large")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f: f.write(data)
    # כתוב מטא
    meta = {"ttl_sec": int(_fs.ttl_sec_default if ttl_sec is None else ttl_sec), "created": time.time()}
    with open(full + ".meta.json", "w", encoding="utf-8") as f: json.dump(meta, f)

def read_text(rel_path: str) -> str:
    full = _safe_join(rel_path)
    if not os.path.exists(full): raise FSPathError("not_found")
    # בדיקת TTL
    mpath = full + ".meta.json"
    if os.path.exists(mpath):
        with open(mpath,"r",encoding="utf-8") as f: meta = json.load(f)
        ttl = int(meta.get("ttl_sec",0)); created=float(meta.get("created",0))
        if ttl>0 and (time.time()-created)>ttl:
            # expire
            try:
                os.remove(full); os.remove(mpath)
            except OSError:
                pass
            raise FSPathError("expired")
    with open(full, "rb") as f: return f.read().decode("utf-8", "replace")

def delete_path(rel_path: str):
    full = _safe_join(rel_path)
    if os.path.isdir(full):
        shutil.rmtree(full, ignore_errors=True)
    else:
        for suf in ("", ".meta.json"):
            try: os.remove(full + (suf if suf==".meta.json" else ""))
            except OSError: pass


class FSSandbox:
    """
    File-system sandbox with a fixed base directory.
    - Prevents path traversal outside base (..).
    - Optional read-only mode.
    """

    def __init__(self, base: str, readonly: bool = True):
        self.base = os.path.realpath(base)
        os.makedirs(self.base, exist_ok=True)
        self.readonly = readonly

    def _safe_path(self, rel: str) -> str:
        if rel.startswith("/"):
            raise FSError("absolute_path_not_allowed")
        path = os.path.realpath(os.path.join(self.base, rel))
        if not path.startswith(self.base + os.sep) and path != self.base:
            raise FSError("path_traversal_blocked")
        return path

    def read_text(self, rel: str, encoding: str = "utf-8") -> str:
        path = self._safe_path(rel)
        if not os.path.exists(path):
            raise FSError("file_not_found")
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def write_text(self, rel: str, content: str, encoding: str = "utf-8") -> None:
        if self.readonly:
            raise FSError("readonly")
        path = self._safe_path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)

    def exists(self, rel: str) -> bool:
        path = self._safe_path(rel)
        return os.path.exists(path)

    def list(self, rel: str = ".") -> list[str]:
        path = self._safe_path(rel)
        if not os.path.isdir(path):
            raise FSError("not_a_directory")
        return sorted(os.listdir(path))
