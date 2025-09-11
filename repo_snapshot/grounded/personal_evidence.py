# imu_repo/grounded/personal_evidence.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os, json, time, hashlib
from grounded.ttl import TTLPolicy

class PersonalProvenance:
    """
    אחסון ראיות פר־משתמש עם חתימת תוכן (sha256) וסימון _sig_ok/_fresh בהתאם ל-TTL.
    """
    def __init__(self, root: str="/mnt/data/imu_repo/users"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _dir(self, user_id: str) -> str:
        p = os.path.join(self.root, user_id, "evidence")
        os.makedirs(p, exist_ok=True)
        return p

    def put(self, user_id: str, key: str, payload: Dict[str,Any], *,
            kind: str, confidence: float, trust: float,
            seen_count: int=1, stable: bool=False, source_url: str="user://self") -> Dict[str,Any]:
        expire_ts = TTLPolicy.compute_expire_ts(kind, confidence=confidence, seen_count=seen_count, stable=stable)
        rec = {
            "_id": f"{key}:{int(time.time()*1e6)}",
            "key": key,
            "kind": kind,
            "payload": payload,
            "confidence": float(confidence),
            "trust": float(trust),
            "source_url": source_url,
            "expire_ts": expire_ts,
        }
        # חתימה
        h = hashlib.sha256()
        h.update(json.dumps({"key":key,"payload":payload,"ts":int(time.time())}, sort_keys=True).encode("utf-8"))
        rec["_sha256"] = h.hexdigest()
        rec["_sig_ok"] = True
        rec["_fresh"] = TTLPolicy.is_fresh(expire_ts)
        # כתיבה
        dp = self._dir(user_id)
        with open(os.path.join(dp, f"{rec['_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec

    def scan(self, user_id: str) -> Dict[str,Any]:
        out=[]
        dp = self._dir(user_id)
        for name in os.listdir(dp):
            if not name.endswith(".json"): continue
            try:
                with open(os.path.join(dp,name),"r",encoding="utf-8") as f:
                    rec=json.load(f)
                rec["_fresh"] = TTLPolicy.is_fresh(rec.get("expire_ts"))
                out.append(rec)
            except: pass
        return {"records": out}