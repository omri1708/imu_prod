# audit/merkle_log.py
from __future__ import annotations
import os, json, hashlib, time, threading
from typing import Any, Dict, Optional

class MerkleAudit:
    def __init__(self, base: str):
        self.base = base
        os.makedirs(base, exist_ok=True)
        self._lock = threading.RLock()
        self._idx_path = os.path.join(base, "index.json")
        if not os.path.exists(self._idx_path):
            with open(self._idx_path, "w", encoding="utf-8") as f:
                json.dump({"seq":0, "root": None}, f)

    def _h(self, b: bytes) -> str: return hashlib.sha256(b).hexdigest()

    def append(self, topic: str, event: Dict[str,Any]) -> Dict[str,Any]:
        with self._lock:
            idx = json.loads(open(self._idx_path,"r",encoding="utf-8").read())
            seq = int(idx.get("seq",0)) + 1
            prev = idx.get("root")
            rec = {"seq": seq, "ts": time.time(), "topic": topic, "event": event, "prev": prev}
            blob = json.dumps(rec, ensure_ascii=False).encode("utf-8")
            h = self._h(blob)
            rec["hash"] = h
            # עדכון root = H(prev||hash)
            root_blob = (prev or "").encode("utf-8") + h.encode("utf-8")
            root = self._h(root_blob)
            rec["root"] = root
            with open(os.path.join(self.base, f"{seq:020d}.json"),"w",encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            with open(self._idx_path,"w",encoding="utf-8") as f:
                json.dump({"seq": seq, "root": root}, f)
            return {"seq": seq, "root": root, "hash": h}

    def verify(self) -> bool:
        prev = None
        for fn in sorted([f for f in os.listdir(self.base) if f.endswith('.json') and f != 'index.json']):
            rec = json.loads(open(os.path.join(self.base, fn),"r",encoding="utf-8").read())
            blob = json.dumps({k:v for k,v in rec.items() if k not in ("hash","root")}, ensure_ascii=False).encode("utf-8")
            if self._h(blob) != rec.get("hash"): return False
            root_blob = (prev or "").encode("utf-8") + rec["hash"].encode("utf-8")
            if self._h(root_blob) != rec.get("root"): return False
            prev = rec["root"]
        return True