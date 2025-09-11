# imu_repo/grounded/ttl.py
from __future__ import annotations
from typing import Optional
import time
import math

class TTLPolicy:
    """
    TTL דינמי על בסיס confidence, frequency והאם זו העדפה 'יציבה' (stable).
    החישוב מחזיר expire_ts (או None אם אין תפוגה).
    """
    BASE_TTL_S = {
        "preference": 30*24*3600,   # 30 ימים
        "belief":     14*24*3600,   # 14 ימים
        "goal":       7*24*3600,    # 7 ימים
        "emotion":    6*3600,       # 6 שעות
        "context":    24*3600,      # יום
    }

    @classmethod
    def compute_expire_ts(cls, kind: str, *, confidence: float, seen_count: int, stable: bool) -> Optional[float]:
        base = cls.BASE_TTL_S.get(kind, 7*24*3600)
        c = max(0.0, min(1.0, float(confidence)))
        # rule: יותר ביטחון/יותר מופעים → TTL ארוך יותר; stable מכפיל
        mult = 0.5 + 1.5*c + 0.1*math.log1p(max(0, seen_count))
        if stable: mult *= 2.0
        ttl = base * mult
        return time.time() + ttl

    @classmethod
    def is_fresh(cls, expire_ts: Optional[float]) -> bool:
        return (expire_ts is None) or (expire_ts > time.time())