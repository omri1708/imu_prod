# engine/llm/bandit_selector.py
from __future__ import annotations
import math, random
from typing import Dict, Any, List, Tuple

class UCBSelector:
    """בחירת תצורת מודל/פרומפט/טמפרטורה לפי UCB1 על reward (דיוק/latency/עלות משוקללים)."""
    def __init__(self, arms: List[Dict[str,Any]]):
        self.arms = arms
        self.N = [1e-9]*len(arms)
        self.R = [0.0]*len(arms)
        self.t = 0

    def select(self) -> Tuple[int, Dict[str,Any]]:
        self.t += 1
        ucb = [(self.R[i]/max(self.N[i],1e-9)) + math.sqrt(2*math.log(max(self.t,1.0))/max(self.N[i],1e-9)) for i in range(len(self.arms))]
        i = max(range(len(self.arms)), key=lambda j: ucb[j])
        return i, self.arms[i]

    def update(self, i: int, reward: float) -> None:
        self.N[i] += 1
        self.R[i] += reward