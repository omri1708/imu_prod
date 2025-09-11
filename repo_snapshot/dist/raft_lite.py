from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, json, time, uuid, multiprocessing as mp, threading

CLUSTER_ROOT = "/mnt/data/imu_repo/cluster"
HEARTBEAT_INT_S = 0.5
STALE_S = 2.0

def _ensure():
    os.makedirs(CLUSTER_ROOT, exist_ok=True)

def _node_dir(node_id: str) -> str:
    return os.path.join(CLUSTER_ROOT, f"node_{node_id}")

def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def _read_json(path: str) -> Any:
    if not os.path.exists(path): return None
    return json.load(open(path, "r", encoding="utf-8"))

def _now() -> float: return time.time()

def _list_nodes() -> List[str]:
    _ensure()
    out=[]
    for n in os.listdir(CLUSTER_ROOT):
        if n.startswith("node_"):
            out.append(n.split("_",1)[1])
    return sorted(out)

def _leader_file() -> str:
    return os.path.join(CLUSTER_ROOT, "leader.json")

def current_leader() -> Optional[str]:
    l = _read_json(_leader_file()) or {}
    ts = l.get("ts", 0.0)
    if _now() - ts > STALE_S:
        return None
    return l.get("id")

def _set_leader(node_id: str) -> None:
    _write_json(_leader_file(), {"id": node_id, "ts": _now()})

def _heartbeat_path(node_id: str) -> str:
    return os.path.join(_node_dir(node_id), "heartbeat.json")

def _log_path(node_id: str) -> str:
    return os.path.join(_node_dir(node_id), "log.jsonl")

def is_alive(node_id: str) -> bool:
    p = _heartbeat_path(node_id)
    hb = _read_json(p) or {}
    ts = hb.get("ts", 0.0)
    return (_now() - ts) <= STALE_S

def _append_log(node_id: str, rec: Dict[str,Any]) -> None:
    os.makedirs(_node_dir(node_id), exist_ok=True)
    with open(_log_path(node_id), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), **rec}, ensure_ascii=False) + "\n")

def cluster_health() -> Dict[str,Any]:
    nodes = _list_nodes()
    alive = [n for n in nodes if is_alive(n)]
    return {"nodes": nodes, "alive": alive, "leader": current_leader(),
            "quorum_ok": (len(alive) >= (len(nodes)//2 + 1) if nodes else False)}

class Node(mp.Process):
    """
    Raft-Lite:
      - כל תהליך כותב heartbeat לקובץ node_{id}/heartbeat.json.
      - אם אין מנהיג או שהוא ישן => בחירות: הנוד עם ה-id הקטן ביותר מבין החיים יכריז עצמו כמנהיג.
      - רק המנהיג מוסיף לרשומת הלוג של עצמו (append_record).
    """
    def __init__(self, node_id: str):
        super().__init__(daemon=True)
        self.node_id = node_id
        self._stop = mp.Event()

    def stop(self): self._stop.set()

    def _beat_forever(self):
        nd = _node_dir(self.node_id)
        os.makedirs(nd, exist_ok=True)
        while not self._stop.is_set():
            _write_json(_heartbeat_path(self.node_id), {"ts": _now()})
            # שמירה על מנהיג
            leader = current_leader()
            if leader is None:
                # בחירות: בחר את הקטן ביותר מבין החיים
                alive = [n for n in _list_nodes() if is_alive(n)]
                if alive:
                    cand = sorted(alive)[0]
                    if cand == self.node_id:
                        _set_leader(self.node_id)
                        _append_log(self.node_id, {"type":"election","leader":self.node_id})
            time.sleep(HEARTBEAT_INT_S)

    def run(self):
        _ensure()
        t = threading.Thread(target=self._beat_forever, daemon=True)
        t.start()
        # המתן עד עצירה
        while not self._stop.is_set():
            time.sleep(0.1)

def ensure_node(node_id: Optional[str]=None) -> Node:
    _ensure()
    node_id = node_id or uuid.uuid4().hex[:8]
    os.makedirs(_node_dir(node_id), exist_ok=True)
    n = Node(node_id)
    n.start()
    return n

def append_record_if_leader(record: Dict[str,Any]) -> bool:
    leader = current_leader()
    if not leader or not is_alive(leader):
        return False
    _append_log(leader, {"type":"record", "payload": record})
    # עדכן חותמת למניעת התיישנות
    _set_leader(leader)
    return True