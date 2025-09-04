# imu_repo/obs/tracing.py
from __future__ import annotations
import time, uuid, json, os
from typing import Dict, Any, List, Optional

class Span:
    def __init__(self, name: str, parent_id: Optional[str] = None):
        self.span_id = uuid.uuid4().hex
        self.parent_id = parent_id
        self.name = name
        self.ts_start = time.time()
        self.ts_end: float | None = None
        self.attrs: Dict[str, Any] = {}

    def set_attr(self, k: str, v: Any):
        self.attrs[k] = v

    def end(self):
        self.ts_end = time.time()

class Tracer:
    def __init__(self, path: str = ".imu_state/trace.jsonl"):
        self.spans: List[Span] = []
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path,"w",encoding="utf-8"): pass
    
    def emit(self, event: str, data: Dict[str,Any]):
        rec={"ts": time.time(), "event": event, "data": data}
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")    

    def start_span(self, name: str, parent_id: Optional[str] = None) -> Span:
        s = Span(name, parent_id)
        self.spans.append(s)
        return s

    def end_span(self, span: Span):
        span.end()

    def export(self) -> List[Dict[str,Any]]:
        out=[]
        for s in self.spans:
            out.append({
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "ts_start": s.ts_start,
                "ts_end": s.ts_end or time.time(),
                "attrs": s.attrs
            })
        return out
