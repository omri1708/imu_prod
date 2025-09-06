# runtime/p95.py
from __future__ import annotations
from typing import List, Dict
import time, bisect

class P95Window:
    def __init__(self, max_samples: int = 512):
        self.values: List[float] = []
        self.max_samples = max_samples

    def add(self, ms: float):
        bisect.insort(self.values, ms)
        if len(self.values) > self.max_samples:
            del self.values[0]

    def p95(self) -> float:
        if not self.values: return 0.0
        idx = int(0.95*(len(self.values)-1))
        return self.values[idx]

class P95Gates:
    def __init__(self):
        self.windows: Dict[str, P95Window] = {}
    def observe(self, key: str, ms: float):
        self.windows.setdefault(key, P95Window()).add(ms)
    def ensure(self, key: str, ceiling_ms: int):
        p95 = self.windows.get(key).p95() if key in self.windows else 0.0
        if p95 > ceiling_ms:
            raise RuntimeError(f"p95_exceeded: key={key} p95={p95:.1f}ms ceiling={ceiling_ms}ms")

GATES = P95Gates()