# imu_repo/dist/crdt.py
from __future__ import annotations
from typing import Dict, Any, Set, Tuple, Optional
import time

Clock = float  # wall-clock; ל-demo. במערכות אמת מומלץ hybrid logical clock.

def now() -> Clock: return time.time()

class LWWRegister:
    def __init__(self, value: Any=None, ts: Clock=0.0): 
        self.value, self.ts = value, ts
    def set(self, value: Any, ts: Optional[Clock]=None):
        t = now() if ts is None else float(ts)
        if t >= self.ts:
            self.value, self.ts = value, t
    def merge(self, other: "LWWRegister"):
        if other.ts > self.ts:
            self.value, self.ts = other.value, other.ts

class LWWSet:
    def __init__(self): 
        self.adds: Dict[Any,Clock] = {}
        self.rems: Dict[Any,Clock] = {}
    def add(self, x: Any, ts: Optional[Clock]=None):
        t = now() if ts is None else float(ts)
        self.adds[x] = max(t, self.adds.get(x, 0.0))
    def remove(self, x: Any, ts: Optional[Clock]=None):
        t = now() if ts is None else float(ts)
        self.rems[x] = max(t, self.rems.get(x, 0.0))
    def value(self) -> Set[Any]:
        out=set()
        for k,ta in self.adds.items():
            if ta > self.rems.get(k, -1.0):
                out.add(k)
        return out
    def merge(self, other: "LWWSet"):
        for k, t in other.adds.items():
            self.adds[k] = max(t, self.adds.get(k, 0.0))
        for k, t in other.rems.items():
            self.rems[k] = max(t, self.rems.get(k, 0.0))

class ORMap:
    """
    OR-Map שמכיל registers/sets לכל מפתח.
    """
    def __init__(self):
        self.keys = LWWSet()
        self.regs: Dict[str, LWWRegister] = {}
        self.sets: Dict[str, LWWSet] = {}

    def put_reg(self, key: str, value: Any, ts: Optional[Clock]=None):
        self.keys.add(key, ts)
        r = self.regs.get(key) or LWWRegister()
        r.set(value, ts)
        self.regs[key] = r

    def get_reg(self, key: str) -> Any:
        if key not in self.keys.value(): 
            return None
        return (self.regs.get(key) or LWWRegister()).value

    def upd_set_add(self, key: str, val: Any, ts: Optional[Clock]=None):
        self.keys.add(key, ts)
        s = self.sets.get(key) or LWWSet()
        s.add(val, ts); self.sets[key]=s

    def upd_set_rem(self, key: str, val: Any, ts: Optional[Clock]=None):
        self.keys.add(key, ts)
        s = self.sets.get(key) or LWWSet()
        s.remove(val, ts); self.sets[key]=s

    def get_set(self, key: str):
        if key not in self.keys.value(): return set()
        return (self.sets.get(key) or LWWSet()).value()

    def merge(self, other: "ORMap"):
        self.keys.merge(other.keys)
        for k, r in other.regs.items():
            lr = self.regs.get(k) or LWWRegister()
            lr.merge(r)
            self.regs[k] = lr
        for k, s in other.sets.items():
            ls = self.sets.get(k) or LWWSet()
            ls.merge(s)
            self.sets[k] = ls