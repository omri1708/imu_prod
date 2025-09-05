# imu_repo/engine/canary_controller.py
from __future__ import annotations
import time, math
from typing import List, Dict, Any, Optional

class CanaryError(Exception): ...

class CanaryStage:
    __slots__ = ("name","percent","min_hold_sec")
    def __init__(self, name: str, percent: int, min_hold_sec: int):
        self.name = str(name); self.percent = int(percent); self.min_hold_sec = int(min_hold_sec)

class CanaryPlan:
    def __init__(self, stages: List[CanaryStage]):
        if not stages: raise CanaryError("empty canary plan")
        ps = [s.percent for s in stages]
        if sorted(ps) != ps:
            raise CanaryError("stages must be non-decreasing percent")
        self.stages = stages

class CanaryRun:
    def __init__(self, plan: CanaryPlan, *, backoff_base_sec: int = 5, max_backoff_sec: int = 300):
        self.plan = plan
        self.idx = 0
        self.started_at = time.time()
        self.stage_started_at = self.started_at
        self.failures = 0
        self.backoff_base = int(backoff_base_sec)
        self.max_backoff = int(max_backoff_sec)
        self.aborted = False
        self.completed = False

    def current(self) -> CanaryStage:
        return self.plan.stages[self.idx]

    def status(self) -> Dict[str,Any]:
        return {
            "idx": self.idx,
            "stage": {"name": self.current().name, "percent": self.current().percent},
            "failures": self.failures,
            "aborted": self.aborted,
            "completed": self.completed
        }

    def _backoff_sleep_sec(self) -> int:
        if self.failures <= 0: return 0
        return min(self.max_backoff, int(self.backoff_base * (2 ** (self.failures - 1))))

    def allow_advance(self) -> bool:
        return (time.time() - self.stage_started_at) >= self.current().min_hold_sec

    def on_gate_pass(self) -> Dict[str,Any]:
        if self.aborted or self.completed:
            return self.status()
        if not self.allow_advance():
            return self.status()
        if self.idx >= len(self.plan.stages) - 1:
            self.completed = True
            return self.status()
        self.idx += 1
        self.stage_started_at = time.time()
        return self.status()

    def on_gate_fail(self, *, hard_abort: bool=False) -> Dict[str,Any]:
        self.failures += 1
        if hard_abort or self.idx == 0:
            self.aborted = True
            return self.status()
        # רולבאק שלב אחד, החזקת backoff לפני הניסיון הבא
        self.idx -= 1
        self.stage_started_at = time.time() + self._backoff_sleep_sec()
        return self.status()