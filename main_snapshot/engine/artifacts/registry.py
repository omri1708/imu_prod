# PATH: engine/artifacts/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import json, hashlib, os

ROOT = Path(os.getenv("IMU_ARTIFACTS_DIR", ".imu/artifacts"))
ROOT.mkdir(parents=True, exist_ok=True)

def _digest(files: Dict[str, bytes|str]) -> str:
    h = hashlib.sha256()
    for k in sorted(files.keys()):
        v = files[k]
        if isinstance(v, str):
            v = v.encode("utf-8")
        h.update(k.encode())
        h.update(b"\0")
        h.update(v)
    return h.hexdigest()

def register(name: str, files: Dict[str, bytes|str]) -> str:
    dg = _digest(files)
    d = ROOT / name / dg
    d.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        p.write_bytes(data)
    (ROOT / name / "latest.json").write_text(json.dumps({"digest": dg, "files": list(files.keys())}, indent=2, ensure_ascii=False), "utf-8")
    return dg

def latest(name: str) -> Dict[str, Any]:
    p = ROOT / name / "latest.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}
