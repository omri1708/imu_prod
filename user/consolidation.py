# imu_repo/user/consolidation.py
from __future__ import annotations
import time, math, threading
from typing import Dict, Any, List, Optional
from user.memory_state import MemoryState
from user.consciousness import UserConsciousness

from user.memory_state import MemoryState


class Consolidation:
    """
    מנהל קונסולידציה:
    - Promote: T1 → T2 כשמידע יציב לאורך זמן/שימושים.
    - Demote/Expire: לפי TTL.
    - Cross-session learning: סיכום אינטראקציות לעדכון תודעה.
    """

    def __init__(self, mem: MemoryState, ucon: UserConsciousness):
        self.mem = mem
        self.ucon = ucon

    def _stable(self, key: str, min_age: float = 24*3600, min_hits: int = 2) -> bool:
        rec = self.mem._load(self.mem.t1_file).get(key)
        if not rec: return False
        age = time.time() - rec.get("ts", 0)
        hits = rec.get("hits", 1)
        return age >= min_age and hits >= min_hits

    def observe(self, key: str):
        # העלאת counter לשימושים — ניצול בקונסולידציה
        d = self.mem._load(self.mem.t1_file)
        if key in d:
            rec = d[key]; rec["hits"] = int(rec.get("hits", 0)) + 1; rec["ts"] = time.time()
            self.mem._save(self.mem.t1_file, d)

    def consolidate(self):
        t1 = self.mem._load(self.mem.t1_file)
        t2 = self.mem._load(self.mem.t2_file)
        changed = False
        for k, rec in list(t1.items()):
            if self._stable(k):
                t2[k] = rec; del t1[k]; changed = True
        if changed:
            self.mem._save(self.mem.t1_file, t1)
            self.mem._save(self.mem.t2_file, t2)

    def on_interaction(self, user_id: str, text: str, derived_preferences: Dict[str,Any] | None = None):
        """נקרא בסוף ריצה: עדכון זיכרון ותודעה."""
        self.mem.remember("__last_text__", text, tier=1, ttl=7*24*3600)
        if derived_preferences:
            for k, v in derived_preferences.items():
                self.mem.remember(f"pref:{k}", v, tier=1, ttl=30*24*3600)
                self.observe(f"pref:{k}")
        self.consolidate()

        # למידה סמנטית ארוכת טווח
        try:
            self.ucon.grant_consent(user_id)  # אם כבר ניתן — לא מזיק
        except Exception:
            pass
        self.ucon.semantic_learn(user_id, text)
        if derived_preferences:
            for k, v in derived_preferences.items():
                self.ucon.update_belief(user_id, f"pref:{k}", v, trust=0.65, ttl=90*24*3600)
        self.ucon.resolve_conflicts(user_id)


class Consolidator:
    """
    מחזיק לולאת איחוד רקע (אופציונלי). ניתן להפעיל מתסריט/דשבורד.
    """
    def __init__(self, mem: MemoryState, period_s: float = 10.0):
        self.mem=mem; self.period=period_s
        self._stop=False; self._thr: Optional[threading.Thread]=None

    def start(self):
        if self._thr and self._thr.is_alive(): return
        self._stop=False
        self._thr=threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop=True
        if self._thr: self._thr.join(timeout=2.0)

    def _loop(self):
        while not self._stop:
            self.mem.consolidate()
            self.mem.decay()
            time.sleep(self.period)