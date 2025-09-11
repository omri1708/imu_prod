# imu_repo/governance/proof_of_convergence.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import statistics, json, os, time, hashlib

class ConvergenceTracker:
    """
    עוקב אחר Φ לאורך חלון, מחשב שיפוע/וריאנס/מונוטוניות, ומחזיר מצב: converging / diverging / undecided.
    """
    def __init__(self, window:int=10, epsilon:float=0.01, max_violations:int=2):
        self.window=window
        self.epsilon=epsilon
        self.max_viol=max_violations
        self.series: List[float] = []

    def add(self, phi: float):
        self.series.append(float(phi))
        if len(self.series) > self.window:
            self.series.pop(0)

    def _slope(self) -> float:
        n=len(self.series)
        if n<2: return 0.0
        xs=list(range(n)); ys=self.series
        xbar=sum(xs)/n; ybar=sum(ys)/n
        num=sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))
        den=sum((x-xbar)**2 for x in xs) or 1.0
        return num/den

    def status(self) -> Dict[str,Any]:
        n=len(self.series)
        if n<3:
            return {"state":"undecided","count":n,"last":self.series[-1] if self.series else None}
        slope=self._slope()
        var=statistics.pvariance(self.series) if n>1 else 0.0
        viol=sum(1 for i in range(1,n) if self.series[i] > self.series[i-1] + 1e-12)
        if slope <= -self.epsilon and viol <= self.max_viol:
            return {"state":"converging","slope":slope,"variance":var,"violations":viol,"last":self.series[-1]}
        if slope >= self.epsilon and viol >= self.max_viol:
            return {"state":"diverging","slope":slope,"variance":var,"violations":viol,"last":self.series[-1]}
        return {"state":"undecided","slope":slope,"variance":var,"violations":viol,"last":self.series[-1]}

class SafeProgressLedger:
    """
    ספר-חשבון בלתי-נמחק (append-only) עם שרשור hash (כמו בלוקצ'יין פרטי) עבור החלטות אימוץ/דחיה.
    """
    def __init__(self, path:str=".imu_state/ledger.jsonl"):
        self.path=path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path,"w",encoding="utf-8"): pass

    def _last_hash(self) -> Optional[str]:
        last=None
        with open(self.path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj=json.loads(line)
                last=obj.get("hash")
        return last

    def append(self, event: Dict[str,Any]) -> str:
        prev=self._last_hash() or ""
        blob=json.dumps({"prev":prev,"event":event}, ensure_ascii=False, sort_keys=True).encode()
        h=hashlib.sha256(blob).hexdigest()
        rec={"prev":prev,"event":event,"hash":h,"ts":time.time()}
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        return h
