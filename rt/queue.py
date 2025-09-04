# imu_repo/rt/queue.py
from __future__ import annotations
import os, json, time, threading, uuid
from typing import Optional, Dict, Any, List, Tuple

class DurableQueue:
    """
    תור עמיד: journal JSONL + קובץ מצב קטן ל-inflight.
    - put(payload) -> msg_id
    - get() -> (msg_id, payload)  (לא מוציא מהיומן עד ack/nack)
    - ack(msg_id)
    - nack(msg_id)  (יחזיר לתור)
    - requeue_stale(inflight_ttl_s)  (מחזיר הודעות שנתקעו)
    """
    def __init__(self, root: str = "/mnt/data/imu_repo/rtq", name: str = "main"):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.journal = os.path.join(self.dir, "journal.jsonl")
        self.state = os.path.join(self.dir, "state.json")
        self._lock = threading.RLock()
        if not os.path.exists(self.state):
            with open(self.state, "w", encoding="utf-8") as f:
                json.dump({"cursor": 0, "inflight": {}}, f)

    def _load_state(self) -> Dict[str, Any]:
        with open(self.state, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, st: Dict[str, Any]) -> None:
        tmp = self.state + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, self.state)

    def put(self, payload: Dict[str, Any]) -> str:
        mid = str(uuid.uuid4())
        rec = {"id": mid, "ts": time.time(), "payload": payload}
        with self._lock, open(self.journal, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return mid

    def get(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            st = self._load_state()
            cur = int(st.get("cursor", 0))
            # קרא את השורה ה-cur
            if not os.path.exists(self.journal): 
                return None
            with open(self.journal, "r", encoding="utf-8") as f:
                for i, ln in enumerate(f):
                    if i < cur: 
                        continue
                    try:
                        rec = json.loads(ln)
                    except:
                        continue
                    st["cursor"] = i + 1
                    st["inflight"][rec["id"]] = {"ts": time.time(), "payload": rec["payload"], "idx": i}
                    self._save_state(st)
                    return rec["id"], rec["payload"]
        return None

    def ack(self, msg_id: str) -> None:
        with self._lock:
            st = self._load_state()
            st["inflight"].pop(msg_id, None)
            self._save_state(st)

    def nack(self, msg_id: str) -> None:
        with self._lock:
            st = self._load_state()
            info = st["inflight"].pop(msg_id, None)
            if info is not None:
                # מחזיר את ההודעה ע"י הכנסת רשומה חדשה לסוף
                self.put(info["payload"])
            self._save_state(st)

    def requeue_stale(self, inflight_ttl_s: float = 30.0) -> int:
        """מחזיר הודעות שנתקעו יותר מ-inflight_ttl_s."""
        now = time.time()
        moved = 0
        with self._lock:
            st = self._load_state()
            stale = [mid for mid, inf in st["inflight"].items() if (now - inf["ts"]) > inflight_ttl_s]
            for mid in stale:
                self.put(st["inflight"][mid]["payload"])
                st["inflight"].pop(mid, None)
                moved += 1
            self._save_state(st)
        return moved