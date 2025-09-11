# adapters/contracts/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import shutil, os, subprocess, json, time, hashlib, pathlib, threading, queue
from datetime import datetime, timezone

class ContractError(Exception): ...

class ResourceRequired(ContractError):
    def __init__(self, resource: str, how_to_install: str, why: str):
        super().__init__(f"resource_required: {resource} | {why} | install: {how_to_install}")
        self.resource = resource; self.how_to_install = how_to_install; self.why = why

class ProcessFailed(ContractError):
    def __init__(self, cmd: List[str], rc: int, out: str, err: str):
        super().__init__(f"process_failed rc={rc} cmd={' '.join(cmd)}"); self.cmd=cmd; self.rc=rc; self.out=out; self.err=err

def require_binary(name: str, how_to: str, why: str):
    if shutil.which(name) is None:
        raise ResourceRequired(resource=name, how_to_install=how_to, why=why)

def run(cmd: List[str], cwd: Optional[str]=None, env: Optional[Dict[str,str]]=None, timeout: Optional[int]=None) -> str:
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); raise ProcessFailed(cmd, -9, "", "timeout")
    if p.returncode != 0:
        raise ProcessFailed(cmd, p.returncode, out, err)
    return out

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

@dataclass
class BuildResult:
    artifact: str
    sha256: str
    meta: Dict[str, Any]

# ---- Provenance / CAS / Audit ----
class CAS:
    def __init__(self, root: str = ".imu_cas"):
        self.root = root; ensure_dir(self.root)
    def put_file(self, path: str) -> str:
        digest = sha256_file(path)
        dst = os.path.join(self.root, digest)
        if not os.path.exists(dst): shutil.copy2(path, dst)
        return digest
    def has(self, digest: str) -> bool: return os.path.exists(os.path.join(self.root, digest))
    def path(self, digest: str) -> str: return os.path.join(self.root, digest)

class AuditLog:
    def __init__(self, path: str = ".imu_audit.jsonl"):
        self.path = path; ensure_dir(os.path.dirname(path) or ".")
        self._lock = threading.Lock()
    def write(self, entry: Dict[str, Any]):
        entry = dict(entry); entry["ts"]=datetime.now(timezone.utc).isoformat()
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock: 
            with open(self.path, "a", encoding="utf-8") as f: f.write(line+"\n")

AUDIT = AuditLog(".imu/audit.jsonl")
CAS_STORE = CAS(".imu/cas")

def record_event(kind: str, data: Dict[str, Any]):
    AUDIT.write({"kind": kind, **data})