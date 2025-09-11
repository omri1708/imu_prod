# imu_repo/engine/audit_rollup.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, List
from engine.cas_store import put_json
from engine.cas_sign import sign_manifest

class AuditRollupError(Exception): ...

def _audit_dir() -> str:
    d = os.environ.get("IMU_AUDIT_DIR") or ".audit"
    os.makedirs(d, exist_ok=True)
    return d

def _lines_in_window(ts_start: float, ts_end: float) -> List[str]:
    out: List[str] = []
    d = _audit_dir()
    for name in os.listdir(d):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(d, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: 
                        continue
                    try:
                        obj = json.loads(ln)
                        ts = float(obj.get("ts", 0))
                        if ts_start <= ts < ts_end:
                            out.append(ln)
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return out

def _merkle_like_root(lines: List[str]) -> str:
    """מרקל לייט: גיבוב שכבות בזוגות עד לשורש (hex)."""
    if not lines:
        return hashlib.sha256(b"").hexdigest()
    layer = [hashlib.sha256(ln.encode("utf-8")).digest() for ln in lines]
    while len(layer) > 1:
        nxt = []
        it = iter(layer)
        for a in it:
            b = next(it, a)  # אם אי-זוגי — שכפל אחרון
            nxt.append(hashlib.sha256(a + b).digest())
        layer = nxt
    return layer[0].hex()

def rollup_window(*, window_seconds: int = 3600, signing_key: Dict[str,Any] | None=None) -> Dict[str,Any]:
    now = time.time()
    start = now - (now % window_seconds)
    end = start + window_seconds
    lines = _lines_in_window(start, end)
    root = _merkle_like_root(lines)
    bundle = {
        "version": 1,
        "window": {"start": start, "end": end, "secs": window_seconds},
        "count": len(lines),
        "root": root
    }
    if signing_key:
        kid, meta = next(iter(signing_key.items()))
        sig = sign_manifest(bundle, key_id=kid, secret_hex=str(meta["secret_hex"]), algo=str(meta.get("algo","sha256")))
        bundle["signature"] = sig
    rollup_hash = put_json(bundle)
    return {"ok": True, "rollup_hash": rollup_hash, "rollup": bundle}