# synth/verify.py
from __future__ import annotations
import json, hashlib, time, os
from typing import Dict, Any, List

def _sha(b: bytes)->str: return hashlib.sha256(b).hexdigest()

class EvidenceStore:
    def __init__(self, root: str=".imu_state/evidence"):
        self.root=root; os.makedirs(root, exist_ok=True)

    def put(self, claim: str, obj: Dict[str,Any]) -> Dict[str,Any]:
        b=json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
        h=_sha(b); p=os.path.join(self.root, h+".json")
        with open(p,"wb") as f: f.write(b)
        rec={"claim":claim,"hash":h,"path":p,"ts":time.time()}
        return rec

def verify_against_contracts(contracts: List[Dict[str,Any]], test_result: Dict[str,Any]) -> Dict[str,Any]:
    """
    Minimal concrete verification: make sure /health=200 and each endpoint had 200.
    """
    ok=True; violations=[]
    # health contract
    health = any(c.get("name")=="health_ok" for c in contracts)
    if health:
        h = [r for r in test_result["results"] if r["path"]=="/health"]
        if not h or not h[0]["ok"]:
            ok=False; violations.append("health_not_ok")
    # status contract
    all_ok = all(r["ok"] for r in test_result["results"])
    if not all_ok:
        ok=False; violations.append("endpoint_failed")
    return {"ok":ok,"violations":violations}

