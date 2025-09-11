# imu_repo/engine/quarantine.py
from __future__ import annotations
from typing import Dict, Any, Optional
import time

class Quarantined(Exception): ...

class CapabilityGuard:
    """
    שומר מוניטין ליכולות (capability name) עם הסגר (quarantine) על פי:
      - שגיאות/הפרות
      - יחס שגיאה מעל סף
      - backoff גאומטרי
    """
    def __init__(self, *, now=None):
        self._cap: Dict[str, Dict[str, float]] = {}
        self._now = now or (lambda: time.time())

    def before_call(self, cap: str) -> None:
        st = self._cap.get(cap)
        if not st:
            return
        until = st.get("quarantined_until", 0.0)
        if self._now() < until:
            raise Quarantined(f"cap_quarantined:{cap} until {until}")

    def after_call(self, cap: str, *, ok: bool, violations: int = 0, policy: Dict[str,Any]) -> None:
        st = self._cap.setdefault(cap, {
            "calls": 0.0, "errors": 0.0, "violations": 0.0,
            "quarantined_until": 0.0, "backoff_sec": float(policy.get("quarantine_backoff_base_sec", 30.0))
        })
        st["calls"] += 1.0
        if not ok:
            st["errors"] += 1.0
        st["violations"] += float(violations)

        min_calls = int(policy.get("quarantine_min_calls", 20))
        thr_err = float(policy.get("quarantine_error_rate_threshold", 0.2))  # 20%
        thr_vio = float(policy.get("quarantine_violation_rate_threshold", 0.05))  # 5%
        now = self._now()
        if st["calls"] >= max(1.0, float(min_calls)):
            err_rate = st["errors"] / st["calls"]
            vio_rate = st["violations"] / st["calls"]
            if err_rate >= thr_err or vio_rate >= thr_vio:
                # quarantine
                until = now + st["backoff_sec"]
                st["quarantined_until"] = until
                # backoff גאומטרי
                st["backoff_sec"] = min(st["backoff_sec"] * 2.0, float(policy.get("quarantine_backoff_max_sec", 3600.0)))
                # reset counters לאחר הסגר
                st["calls"] = 0.0
                st["errors"] = 0.0
                st["violations"] = 0.0

    def force_release(self, cap: str) -> None:
        if cap in self._cap:
            self._cap[cap]["quarantined_until"] = 0.0