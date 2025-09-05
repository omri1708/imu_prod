# audit/log.py
# -*- coding: utf-8 -*-
import hashlib, json, os, time, threading
from typing import Optional, Dict, Any

class AppendOnlyAudit:
    """לוג מצורף-בלבד עם שרשור hash (prev_hash), כדי להקשות על מחיקות/שינויים שקטים."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            # שיחזור השרשרת מהפעם הקודמת (כדי לא לשבור hash-chain)
            try:
                with open(path, "rb") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.decode("utf-8"))
                            self._last = rec.get("_prev_hash", self._last)
                        except:
                            pass
            except:
                pass
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f: pass

    
    def _tail_hash(self) -> str:
        h = "0"*64
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    h = obj.get("_self_hash", h)
                except: pass
        return h

    
    @staticmethod
    def _hash_line(obj: Dict[str, Any]) -> str:
        s = json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    
    def append(self, record: Dict[str, Any], event: Dict[str, Any], ts: Optional[int] = None)-> None:
        prev = self._tail_hash()
        obj = {"ts": int(ts or time.time()), "event": event, "prev_hash": prev}
        obj["_self_hash"] = self._hash_line(obj)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        with self._lock:
            rec = dict(record)
            rec["_ts"] = int(time.time())
            rec["_prev_hash"] = self._last
            blob = json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")
            h = hashlib.sha256(blob).hexdigest()
            rec["_hash"] = h
            with open(self.path, "ab") as f:
                f.write(json.dumps(rec, ensure_ascii=False).encode("utf-8")+b"\n")
            self._last = h
        
        return obj["_self_hash"]