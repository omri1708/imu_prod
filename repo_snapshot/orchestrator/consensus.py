# imu_repo/orchestrator/consensus.py
from __future__ import annotations
import os, time, uuid

CLUSTER_DIR = "/mnt/data/imu_repo/cluster"
LEADER_FILE = os.path.join(CLUSTER_DIR, "leader.lock")

class LeaderElector:
    """
    בחירת מנהיג ע"י יצירת קובץ אטומית. אם פג תוקף (ttl_s) – מותר 'steal'.
    cross-platform: O_CREAT|O_EXCL (ללא fcntl).
    """
    def __init__(self, node_id: str | None=None, ttl_s: float=8.0):
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self.ttl_s = float(ttl_s)
        os.makedirs(CLUSTER_DIR, exist_ok=True)

    def _write(self, path: str, content: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)

    def try_acquire(self) -> bool:
        now = time.time()
        payload = f"{self.node_id}:{now:.3f}:{self.ttl_s:.3f}"
        try:
            fd = os.open(LEADER_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            try:
                s = open(LEADER_FILE,"r",encoding="utf-8").read().strip()
                parts = s.split(":")
                if len(parts)>=3:
                    other_id = parts[0]
                    ts = float(parts[1]); ttl = float(parts[2])
                    if (now - ts) > ttl:
                        # פג תוקף – נגנוב
                        tmp = LEADER_FILE + ".tmp"
                        self._write(tmp, payload)
                        os.replace(tmp, LEADER_FILE)
                        return True
            except FileNotFoundError:
                # נעלם – ננסה שוב
                return self.try_acquire()
            except Exception:
                # קובץ שבור – החלף
                tmp = LEADER_FILE + ".tmp"
                self._write(tmp, payload)
                os.replace(tmp, LEADER_FILE)
                return True
            return False

    def renew(self) -> None:
        # אם אני לא המנהיג, לא אדרוס
        try:
            s = open(LEADER_FILE,"r",encoding="utf-8").read().strip()
            if s.startswith(self.node_id+":"):
                self._write(LEADER_FILE, f"{self.node_id}:{time.time():.3f}:{self.ttl_s:.3f}")
        except FileNotFoundError:
            pass

    def is_leader(self) -> bool:
        try:
            s = open(LEADER_FILE,"r",encoding="utf-8").read().strip()
            return s.startswith(self.node_id+":")
        except FileNotFoundError:
            return False

    def release(self) -> None:
        try:
            s = open(LEADER_FILE,"r",encoding="utf-8").read().strip()
            if s.startswith(self.node_id+":"):
                os.unlink(LEADER_FILE)
        except FileNotFoundError:
            pass