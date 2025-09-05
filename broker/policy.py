# broker/policy.py — מדיניות השלכה, רמזי עומס, ו-WFQ (חלוקה הוגנת לפי משקל)
# -*- coding: utf-8 -*-
import time, threading, random
from typing import Optional

class DropPolicy:
    TAIL_DROP = "tail_drop"                # זרוק חדש כשמלא
    HEAD_DROP = "head_drop"                # זרוק הישן
    LOWEST_PRIORITY_REPLACE = "lpr"        # החלף את הנמוך ביותר
    RANDOM_EARLY_DROP = "red"              # זריקה הסתברותית

class LoadHint:
    OK = "ok"
    HIGH = "high"
    CRITICAL = "critical"

def load_hint(backlog: int, soft: int, hard: int) -> str:
    if backlog >= hard: return LoadHint.CRITICAL
    if backlog >= soft: return LoadHint.HIGH
    return LoadHint.OK

class WFQ:
    """Weighted Fair Queueing tick counter (פשוט)."""
    def __init__(self):
        self._vtime = 0.0
        self._lock = threading.Lock()
        self._last = time.perf_counter()

    def tick(self, active_weights_sum: float) -> float:
        with self._lock:
            now = time.perf_counter()
            dt = max(0.0, now - self._last)
            self._last = now
            inc = dt / max(1e-6, active_weights_sum)
            self._vtime += inc
            return self._vtime