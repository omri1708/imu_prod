# imu_repo/engine/metrics_watcher.py
from __future__ import annotations
import asyncio, threading, time
from typing import Optional, Dict, Any, Callable

from metrics.aggregate import aggregate_metrics
from engine.hooks import AsyncThrottle

def adapt_once(throttle: AsyncThrottle, *, name: str, window_s: int = 60) -> Dict[str,Any]:
    """
    התאמה חד־פעמית של המצערת לפי מדדי runtime (p95/error/gate_denied) בחלון נתון.
    """
    stats = aggregate_metrics(name=name, bucket=None, window_s=window_s)
    throttle.advise_from_stats(stats)
    return stats

class AdaptiveLoop:
    """
    לולאת התאמה רציפה (thread) שמכוונת את ה-AsyncThrottle לפי מדדי runtime.
    – לא פותחת סוקטים; פועלת על קבצי metrics.jsonl שנצברים.
    """
    def __init__(self, throttle: AsyncThrottle, *, name: str, window_s: int = 60, period_s: float = 2.0):
        self.throttle = throttle
        self.name = name
        self.window_s = int(window_s)
        self.period_s = float(period_s)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self.last_stats: Optional[Dict[str,Any]] = None

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="adaptive_metrics_loop", daemon=True)
        self._thr.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.last_stats = adapt_once(self.throttle, name=self.name, window_s=self.window_s)
            except Exception:
                # לא נכשיל לולאה
                pass
            time.sleep(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=self.period_s * 2.0)