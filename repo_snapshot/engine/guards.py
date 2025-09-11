# engine/guards.py
import time, statistics
from typing import List, Dict
from policy.enforcement import POLICY
from provenance.store import Evidence
from engine.errors import PolicyDenied

class Guard:
    def __init__(self):
        self.latency=[]
    def record_latency(self, ms:int): 
        self.latency.append(ms)
        if len(self.latency)>1000: self.latency=self.latency[-1000:]
    def enforce_p95(self, user:str):
        p95 = int(statistics.quantiles(self.latency, n=20)[18]) if len(self.latency)>=20 else None
        slo = POLICY.get(user).max_p95_ms
        if p95 and p95> slo: 
            raise PolicyDenied(f"p95_exceeded({p95}ms > {slo}ms)")

    def require_evidence(self, user:str, claims:List[Dict], evidences:List[Evidence]):
        if not claims:
            raise PolicyDenied("no_claims_provided")
        if not evidences:
            raise PolicyDenied("no_evidence")
        min_trust = POLICY.get(user).min_evidence_trust
        if any(ev.trust < min_trust for ev in evidences):
            raise PolicyDenied("evidence_trust_below_minimum")

GUARD = Guard()