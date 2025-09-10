from __future__ import annotations
import os, json
from typing import Dict, Any
from audit.merkle_log import MerkleAudit

BASE="var/recovery"; F=os.path.join(BASE,"backoff.json"); os.makedirs(BASE, exist_ok=True)
AUDIT = MerkleAudit("var/audit/pipeline")

def _load(): 
    try: return json.loads(open(F,"r",encoding="utf-8").read())
    except Exception: return {"counters":{}}

def _save(d): 
    with open(F,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)

def allow(key: str, *, attempts_max:int=2) -> Dict[str,Any]:
    d = _load(); c = d.setdefault("counters",{}).get(key,0)
    if c >= attempts_max:
        AUDIT.append("backoff.escalate", {"key":key, "attempts":c})
        return {"ok": False, "escalate": True, "attempts": c}
    d["counters"][key] = c+1; _save(d)
    AUDIT.append("backoff.try", {"key":key, "attempt": c+1})
    return {"ok": True, "attempt": c+1}

def clear(key:str):
    d=_load(); d.get("counters",{}).pop(key,None); _save(d)
    AUDIT.append("backoff.clear", {"key":key})
