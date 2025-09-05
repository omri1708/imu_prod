# imu_repo/engine/reputation.py
from __future__ import annotations
from typing import Dict, Any, Optional
import time
import math

class ReputationRegistry:
    """
    רישום אמון למקורות (source_id) בדעיכה אקספוננציאלית (half-life).
    ניקוד בטווח [-1.0 .. +1.0]. פקטור אמון: 1 + alpha * score  ∈ [1-alpha .. 1+alpha].
    """
    def __init__(self, *, half_life_days: float = 14.0, alpha: float = 0.5, now=None):
        self.half_life_days = float(half_life_days)
        self.alpha = float(alpha)
        self._rep: Dict[str, Dict[str, float]] = {}
        self._now = now or (lambda: time.time())

    def _decay(self, score: float, last_ts: float) -> float:
        dt = max(0.0, self._now() - last_ts)
        if self.half_life_days <= 0:
            return score
        half = self.half_life_days * 86400.0
        # score(t) = score * 0.5^(dt/half)
        return score * math.pow(0.5, dt / half)

    def get_score(self, source_id: str) -> float:
        rec = self._rep.get(source_id)
        if not rec:
            return 0.0
        s = self._decay(rec["score"], rec["ts"])
        # עדכון עצל: מאפסן את הדעיכה כמצב נוכחי
        self._rep[source_id] = {"score": s, "ts": self._now()}
        return s

    def factor(self, source_id: str) -> float:
        # 1 + alpha * score ∈ [1-alpha, 1+alpha]
        s = max(-1.0, min(1.0, self.get_score(source_id)))
        return 1.0 + self.alpha * s

    def update_on_success(self, source_id: str, weight: float = 0.1) -> None:
        rec = self._rep.get(source_id, {"score": 0.0, "ts": self._now()})
        s = self._decay(rec["score"], rec["ts"])
        s = max(-1.0, min(1.0, s + abs(weight)))
        self._rep[source_id] = {"score": s, "ts": self._now()}

    def update_on_violation(self, source_id: str, weight: float = 0.2) -> None:
        rec = self._rep.get(source_id, {"score": 0.0, "ts": self._now()})
        s = self._decay(rec["score"], rec["ts"])
        s = max(-1.0, min(1.0, s - abs(weight)))
        self._rep[source_id] = {"score": s, "ts": self._now()}


class Reputation:
    """
    רפיוטציה נורמלית סביב 1.0 (למשל 0.5..1.5).
    אפשר להאכיל תצפיות שגיאה/הצלחה פר מקור ולהפיק factor.
    """
    def __init__(self, *, base: float=1.0, min_f: float=0.5, max_f: float=1.5):
        self._base = float(base)
        self._min = float(min_f)
        self._max = float(max_f)
        self._ok: Dict[str,int] = {}
        self._bad: Dict[str,int] = {}

    def observe(self, source_id: str, *, ok: bool) -> None:
        if ok:
            self._ok[source_id] = self._ok.get(source_id, 0) + 1
        else:
            self._bad[source_id] = self._bad.get(source_id, 0) + 1

    def factor(self, source_id: str) -> float:
        ok = self._ok.get(source_id, 0)
        bad = self._bad.get(source_id, 0)
        total = ok + bad
        if total <= 0:
            return self._base
        score = (ok + 1.0) / (total + 2.0)  # smoothing
        f = self._min + (self._max - self._min) * score
        return max(self._min, min(self._max, f))