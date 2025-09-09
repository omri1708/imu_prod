# PATH: engine/cache/merkle_cache.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(os.getenv("IMU_CACHE_DIR", ".imu/cache"))
ROOT.mkdir(parents=True, exist_ok=True)

def _key(blob: Any) -> str:
    data = json.dumps(blob, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def get(namespace: str, payload: Any) -> Optional[Dict[str, Any]]:
    k = _key(payload)
    p = ROOT / namespace / (k + ".json")
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return None

def put(namespace: str, payload: Any, result: Dict[str, Any]) -> str:
    k = _key(payload)
    d = ROOT / namespace
    d.mkdir(parents=True, exist_ok=True)
    (d / (k + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    return k
