# ui_dsl/widgets/streams.py (הרחבת DSL לסטרימים: progress/timeline/events)
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Dict, Any, List

class ProgressBar:
    def __init__(self): self.value = 0
    def on_msg(self, m:Dict[str,Any]):
        v = m.get("pct")
        if isinstance(v,(int,float)): self.value = max(0,min(100,int(v)))

class EventTimeline:
    def __init__(self): self.events: List[Dict[str,Any]] = []
    def on_msg(self, m:Dict[str,Any]): self.events.append(m)

class ClaimsView:
    def __init__(self): self.claims=[]
    def on_msg(self, m:Dict[str,Any]): self.claims = m.get("claims",[])

class PerfView:
    def __init__(self): self.perf={}
    def on_msg(self, m:Dict[str,Any]): self.perf = m.get("perf",{})