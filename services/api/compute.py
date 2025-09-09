# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
"""
Weighted score computation helper.
"""



class WeightedScore:
    def __init__(self, name: str, inputs: List[str], weights: List[float]):
        self.name = name
        self.inputs = list(inputs)
        self.weights = list(weights)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "WeightedScore":
        name = str(spec.get("name") or "score")
        inputs = list(spec.get("inputs") or [])
        weights = list(spec.get("weights") or [1.0] * len(inputs))
        if len(weights) != len(inputs):
            raise ValueError("weights length must match inputs length")
        return cls(name, inputs, weights)

    def eval(self, payload: Dict[str, float]) -> float:
        xs = [float(payload.get(k, 0.0)) for k in self.inputs]
        return sum(x * w for x, w in zip(xs, self.weights))
