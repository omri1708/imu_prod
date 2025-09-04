# imu_repo/adapters/db_localqueue.py
from __future__ import annotations
import os, json, threading, time, uuid
from typing import Optional, Dict, Any, List


class QueueError(Exception): ...

class LocalQueue:
    """
    Simple durable queue:
    - put(item) appends JSON line to disk
    - get() returns next available (in-memory cursor), no deletion
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.idx = os.path.join(root, "index.jsonl")
        if not os.path.exists(self.idx):
            with open(self.idx,"w",encoding="utf-8"):
                pass
        self._lock = threading.Lock()
        self._cursor = 0

    def enqueue(self, topic: str, payload: Dict[str,Any]) -> str:
        tid = str(uuid.uuid4())
        path = os.path.join(self.root, f"{tid}.json")
        with open(path,"w",encoding="utf-8") as f: json.dump({"topic":topic,"payload":payload,"ts":time.time()}, f, ensure_ascii=False)
        with open(self.idx,"a",encoding="utf-8") as f: f.write(json.dumps({"id":tid,"topic":topic,"path":path,"state":"ready"})+"\n")
        return tid

    def _lines(self) -> List[Dict[str,Any]]:
        with open(self.idx,"r",encoding="utf-8") as f: 
            return [json.loads(x) for x in f.read().splitlines() if x.strip()]

    def _write_lines(self, arr: List[Dict[str,Any]]):
        with open(self.idx,"w",encoding="utf-8") as f:
            for r in arr:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")

    def dequeue(self, topic: Optional[str] = None) -> Optional[Dict[str,Any]]:
        arr=self._lines()
        for r in arr:
            if r.get("state")=="ready" and (topic is None or r.get("topic")==topic):
                r["state"]="inflight"; r["lease_ts"]=time.time()
                self._write_lines(arr)
                with open(r["path"],"r",encoding="utf-8") as f: payload=json.load(f)
                return {"id":r["id"],"path":r["path"],"payload":payload}
        return None

    def confirm(self, msg_id: str):
        arr=self._lines()
        for i,r in enumerate(arr):
            if r.get("id")==msg_id:
                # מחיקה
                try: os.remove(r["path"])
                except OSError: pass
                del arr[i]; break
        self._write_lines(arr)

    def abandon(self, msg_id: str):
        arr=self._lines()
        for r in arr:
            if r.get("id")==msg_id and r.get("state")=="inflight":
                r["state"]="ready"; r.pop("lease_ts", None)
        self._write_lines(arr)
    def put(self, item: Dict[str, Any]) -> int:
        line = json.dumps(item, ensure_ascii=False)
        with self._lock, open(self.idx, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return 1

    def get(self) -> Optional[Dict[str, Any]]:
        with self._lock, open(self.idx, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < self._cursor: continue
                self._cursor += 1
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    raise QueueError("corrupted_line")
        return None
