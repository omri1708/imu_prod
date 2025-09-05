# imu_repo/engine/consistency_graph_weighted.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from synth.schema_validate import consistent_numbers

class WeightedConsistencyError(Exception): ...

class WeightedConsistencyGraph:
    """
    כמו ConsistencyGraph, אך לכל claim יש weight (למשל סכום trust/רפיוטציה).
    סתירה נמדדת ע"פ משקל; ניתן להגדיר יחס 'dominates' שמאפשר הכרעה.
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str,Any]] = {}
        self.weight: Dict[str, float] = {}
        self.edges: List[Tuple[str,str,Dict[str,Any]]] = []

    def add_claim(self, node_id: str, claim: Dict[str,Any], *, weight: float=1.0) -> None:
        self.nodes[node_id] = dict(claim)
        self.weight[node_id] = float(weight)

    def relate(self, a: str, b: str, rel: str, **meta) -> None:
        self.edges.append((a,b,{"rel":rel, **meta}))

    def _num(self, node_id: str) -> float:
        v = self.nodes.get(node_id, {}).get("value")
        if isinstance(v, (int,float)):
            return float(v)
        raise WeightedConsistencyError(f"node {node_id} not numeric")

    def check(self) -> None:
        for (a,b,m) in self.edges:
            rel = m["rel"]
            if rel == "equal":
                tol = float(m.get("tol_pct", 0.0))
                if not consistent_numbers(self._num(a), self._num(b), tol):
                    # אם יש סתירה — נבדוק דומיננטיות
                    wa, wb = self.weight.get(a,1.0), self.weight.get(b,1.0)
                    dom = m.get("dominates")  # "a" | "b" | None
                    if dom == "a" and wa >= wb: 
                        continue
                    if dom == "b" and wb >= wa: 
                        continue
                    raise WeightedConsistencyError(f"equal conflict: {a} (w={wa}) vs {b} (w={wb})")
            elif rel == "leq":
                if not (self._num(a) <= self._num(b)):
                    raise WeightedConsistencyError(f"leq conflict: {a}>{b}")
            elif rel == "geq":
                if not (self._num(a) >= self._num(b)):
                    raise WeightedConsistencyError(f"geq conflict: {a}<{b}")
            elif rel == "within":
                tol = float(m.get("tol_pct", 0.0))
                x, y = self._num(a), self._num(b)
                if not (y*(1.0-tol) <= x <= y*(1.0+tol)):
                    raise WeightedConsistencyError(f"within conflict: {a} {x} not within ±{tol*100:.1f}% of {b} {y}")
            else:
                raise WeightedConsistencyError(f"unknown relation {rel}")