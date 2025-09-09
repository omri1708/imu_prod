from __future__ import annotations
from typing import Dict, Any, List

class WeightedScore:
    def __init__(self, name: str, inputs: List[str], weights: List[float]):
        self.name, self.inputs, self.weights = name, list(inputs), list(weights)
        if len(self.inputs) != len(self.weights):
            raise ValueError("weights != inputs length")

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "WeightedScore":
        name = str(spec.get("name") or "score")
        inputs = list(spec.get("inputs") or [])
        weights = list(spec.get("weights") or [1.0]*len(inputs))
        return cls(name, inputs, weights)

    def eval(self, payload: Dict[str, float]) -> float:
        xs = [float(payload.get(k, 0.0)) for k in self.inputs]
        return sum(x*w for x, w in zip(xs, self.weights))
