# imu_repo/audit/ledger.py
from __future__ import annotations
from typing import Dict, Any, Iterator
import os, json, hashlib, time

LEDGER_ROOT = "/mnt/data/imu_repo/audit"
LEDGER_FILE = os.path.join(LEDGER_ROOT, "ledger.jsonl")

def _ensure() -> None:
    os.makedirs(LEDGER_ROOT, exist_ok=True)
    if not os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE,"w",encoding="utf-8") as f: pass

def _canon(entry: Dict[str,Any]) -> str:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)

def _hash_entry(e: Dict[str,Any]) -> str:
    return hashlib.sha256(_canon(e).encode("utf-8")).hexdigest()

def _last_hash() -> str | None:
    h = None
    with open(LEDGER_FILE,"r",encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rec = json.loads(ln)
                h = rec.get("_hash")
    return h

def append(entry: Dict[str,Any]) -> Dict[str,Any]:
    _ensure()
    prev = _last_hash()
    e = dict(entry)
    e["_ts"]   = time.time()
    e["_prev"] = prev
    e["_hash"] = _hash_entry({"_ts":e["_ts"], "_prev":e["_prev"], **entry})
    with open(LEDGER_FILE,"a",encoding="utf-8") as f:
        f.write(_canon(e)+"\n")
    return e

def verify_chain() -> bool:
    _ensure()
    prev = None
    with open(LEDGER_FILE,"r",encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): 
                continue
            e = json.loads(ln)
            expected = _hash_entry({"_ts":e["_ts"], "_prev":e["_prev"], **{k:v for k,v in e.items() if not k.startswith("_")}})
            if e.get("_hash") != expected: 
                return False
            if e.get("_prev") != prev:
                return False
            prev = e.get("_hash")
    return True