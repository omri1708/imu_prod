# engine/snapshots/state_snapshot.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Any, Dict, Optional

ROOT = "var/snapshots"
os.makedirs(ROOT, exist_ok=True)


def save_snapshot(spec: Any, *, ctx: Dict[str,Any], artifacts: Optional[Dict[str,Any]]=None) -> str:
    obj = {"ts": time.time(), "user": ctx.get("user_id","anon"), "spec": spec, "artifacts": artifacts or {}}
    blob = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    fp = os.path.join(ROOT, f"{int(time.time())}_{h[:8]}.jsonl")
    with open(fp,"a",encoding="utf-8") as f: f.write(json.dumps(obj, ensure_ascii=False)+"\n")
    return fp