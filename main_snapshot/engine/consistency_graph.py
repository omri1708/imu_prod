# imu_repo/engine/consistency_graph.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict
from synth.schema_validate import consistent_numbers

class GlobalConsistencyError(Exception): ...

class ConsistencyGraph:
    """
    גרף עקביות חוצה־מודולים:
      - צמתים: claim_id (מלא: "<module>:<id>" אם צריך)
      - קשתות: קשרים: must_equal / within_pct / dominates (<=,>=)
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str,Any]] = {}
        self.edges: List[Tuple[str,str,Dict[str,Any]]] = []

    def add_claim(self, node_id: str, claim: Dict[str,Any]) -> None:
        self.nodes[node_id] = dict(claim)

    def relate_must_equal(self, a: str, b: str, *, tol_pct: float = 0.0) -> None:
        self.edges.append((a, b, {"rel":"equal","tol_pct": float(tol_pct)}))

    def relate_within_pct(self, a: str, b: str, *, tol_pct: float) -> None:
        self.edges.append((a, b, {"rel":"within","tol_pct": float(tol_pct)}))

    def relate_leq(self, a: str, b: str) -> None:
        self.edges.append((a, b, {"rel":"leq"}))

    def relate_geq(self, a: str, b: str) -> None:
        self.edges.append((a, b, {"rel":"geq"}))

    def _num(self, node_id: str) -> float:
        c = self.nodes.get(node_id) or {}
        v = c.get("value")
        if isinstance(v, (int,float)):
            return float(v)
        raise GlobalConsistencyError(f"node {node_id} not numeric")

    def _assert(self, cond: bool, msg: str) -> None:
        if not cond:
            raise GlobalConsistencyError(msg)

    def check(self) -> None:
        for (a,b,meta) in self.edges:
            rel = meta.get("rel")
            if rel == "equal":
                tol = float(meta.get("tol_pct", 0.0))
                av = self._num(a); bv = self._num(b)
                self._assert(
                    consistent_numbers(av, bv, tol),
                    f"inconsistent(equal): {a}={av} vs {b}={bv} tol={tol}"
                )
            elif rel == "within":
                tol = float(meta.get("tol_pct", 0.0))
                av = self._num(a); bv = self._num(b)
                hi = bv * (1.0 + tol); lo = bv * (1.0 - tol)
                self._assert(lo <= av <= hi, f"inconsistent(within): {a}={av} not within ±{tol*100:.1f}% of {b}={bv}")
            elif rel == "leq":
                av = self._num(a); bv = self._num(b)
                self._assert(av <= bv, f"inconsistent(leq): {a}={av} > {b}={bv}")
            elif rel == "geq":
                av = self._num(a); bv = self._num(b)
                self._assert(av >= bv, f"inconsistent(geq): {a}={av} < {b}={bv}")
            else:
                raise GlobalConsistencyError(f"unknown relation: {rel}")