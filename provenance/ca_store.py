# imu_repo/provenance/ca_store.py
from __future__ import annotations
import os, json, hashlib, time
from typing import Optional

def _imu_home() -> str:
    home = os.environ.get("IMU_HOME") or os.path.expanduser("~/.imu")
    os.makedirs(home, exist_ok=True)
    return home

def _dir(name: str) -> str:
    d = os.path.join(_imu_home(), name)
    os.makedirs(d, exist_ok=True)
    return d

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def put_bytes(b: bytes) -> str:
    h = sha256_hex(b)
    path = os.path.join(_dir("cas"), h)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b)
    return f"sha256:{h}"

def get_bytes(uri: str) -> Optional[bytes]:
    if not uri.startswith("sha256:"): return None
    h = uri.split(":",1)[1]
    path = os.path.join(_dir("cas"), h)
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        return f.read()

def put_json(obj) -> str:
    b = json.dumps(obj, ensure_ascii=False, separators=(",",":")).encode("utf-8")
    return put_bytes(b)

def index_append(name: str, rec: dict) -> str:
    p = os.path.join(_dir("indexes"), f"{name}.jsonl")
    rec2 = dict(rec); rec2.setdefault("ts", time.time())
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
    return p