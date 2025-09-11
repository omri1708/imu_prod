# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from typing import Dict

class BernoulliTS:
    def __init__(self):
        self.success: Dict[str,int] = {}
        self.fail: Dict[str,int] = {}
    def choose(self, arms):
        import numpy as np
        best, best_v = None, -1
        for a in arms:
            s = self.success.get(a,1); f = self.fail.get(a,1)
            v = random.betavariate(s, f)
            if v > best_v: best, best_v = a, v
        return best or arms[0]
    def update(self, arm, reward: float):
        if reward > 0: self.success[arm] = self.success.get(arm,1)+1
        else: self.fail[arm] = self.fail.get(arm,1)+1

BANDIT = BernoulliTS()
