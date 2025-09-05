# adapters/contracts.py
# -*- coding: utf-8 -*-
import shutil, os, hashlib, json, time, subprocess
from dataclasses import dataclass

class ResourceRequired(Exception):
    def __init__(self, what: str, how_to_install: str):
        super().__init__(f"{what} required")
        self.what = what
        self.how_to_install = how_to_install

@dataclass
class Provenance:
    kind: str
    meta: dict
    sha256: str
    at: float

def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def record_provenance(kind: str, meta: dict, path: str) -> Provenance:
    s = sha256_path(path) if os.path.exists(path) else hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    return Provenance(kind=kind, meta=meta, sha256=s, at=time.time())

def ensure_tool(name: str, hint_cmd: str):
    if shutil.which(name) is None:
        raise ResourceRequired(name, hint_cmd)

def run(cmd: list, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stdout}")
    return p.stdout