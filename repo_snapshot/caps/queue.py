# imu_repo/caps/queue.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os, json, time, uuid

class FileQueue:
    """
    תור מתמיד מבוסס JSONL עם ACK:
      - enqueue: יוצר קובץ msg-<ts>-<uuid>.json
      - dequeue: בוחר את הקובץ הישן ביותר ומסמן לו .lock
      - ack: מוחק את הקובץ המקורי וה-lock
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def _msg_path(self, name: str) -> str:
        return os.path.join(self.path, name)

    def enqueue(self, payload: Dict[str,Any]) -> str:
        name = f"msg-{int(time.time()*1000)}-{uuid.uuid4().hex}.json"
        p = self._msg_path(name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",",":"))
        return name

    def _list_msgs(self):
        names = [n for n in os.listdir(self.path) if n.startswith("msg-") and n.endswith(".json")]
        names.sort()
        return names

    def dequeue(self) -> Optional[Dict[str,Any]]:
        for n in self._list_msgs():
            p = self._msg_path(n)
            lp = p + ".lock"
            try:
                fd = os.open(lp, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                obj["_msg_name"] = n
                return obj
            except Exception:
                # שחרור lock
                try: os.unlink(lp)
                except FileNotFoundError: pass
                continue
        return None

    def ack(self, msg: Dict[str,Any]) -> None:
        n = msg.get("_msg_name")
        if not n: return
        p = self._msg_path(n)
        lp = p + ".lock"
        try: os.unlink(p)
        except FileNotFoundError: pass
        try: os.unlink(lp)
        except FileNotFoundError: pass
