from __future__ import annotations
import math, random, time
from typing import Dict, Any, List, Tuple, Optional

# ---------- Utils ----------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def kpi_to_reward(k: Dict[str,Any], w_cost=0.3, w_lat=0.3, w_qual=0.4) -> float:
    """
    ממפה KPI → תגמול [0..1].
    אם יש targets: p95/err/cost קטנים יותר = טוב יותר; quality (אם יש) גדול יותר = טוב יותר.
    """
    p95  = float(k.get("p95_ms"   , 2000.0))
    terr = float(k.get("error_rate", 0.05))
    cost = float(k.get("cost_usd" , 0.01))
    qual = float(k.get("quality"  , 0.80))

    tp95 = float(k.get("target_ms"  , 1500.0))
    terr_t= float(k.get("target_err", 0.02))
    tcost = float(k.get("target_cost",0.02))
    r_lat  = clamp01((tp95 - p95) / max(tp95,1.0))         # עד כמה מהר מיעד
    r_err  = clamp01((terr_t - terr) / max(terr_t,1e-6))
    r_cost = clamp01((tcost - cost) / max(tcost,1e-6))
    r_qual = clamp01(qual)                                  # אם קיים; אחרת 0.8 דיפולט

    # מורידים את הרעש: משקל טעות לתוך latency/cost
    r = w_qual*r_qual + w_lat*0.5*(r_lat + r_err) + w_cost*r_cost
    return clamp01(r)

# ---------- בסיס ----------
class BaseOpt:
    def select(self, arms: List[Dict[str,Any]]) -> Tuple[int, Dict[str,Any]]: raise NotImplementedError
    def update(self, i: int, reward: float, *, context: Optional[List[float]]=None) -> None: raise NotImplementedError

# ---------- ε-Greedy ----------
class EpsGreedy(BaseOpt):
    def __init__(self, eps: float = 0.1):
        self.eps=eps; self.N:List[float]=[]; self.R:List[float]=[]
    def _init(self, n:int):
        if not self.N: self.N=[1e-9]*n; self.R=[0.0]*n
    def select(self, arms):
        self._init(len(arms))
        if random.random() < self.eps:
            i = random.randrange(len(arms))
        else:
            i = max(range(len(arms)), key=lambda j: self.R[j]/self.N[j])
        return i, arms[i]
    def update(self, i, reward, **_):
        self.N[i]+=1; self.R[i]+=reward

# ---------- UCB1 ----------
class UCB1(BaseOpt):
    def __init__(self):
        self.N:List[float]=[]; self.R:List[float]=[]; self.t=0
    def _init(self, n:int):
        if not self.N: self.N=[1e-9]*n; self.R=[0.0]*n
    def select(self, arms):
        self._init(len(arms)); self.t+=1
        ucb=[ self.R[i]/self.N[i] + math.sqrt(2*math.log(max(self.t,1.0))/self.N[i]) for i in range(len(arms)) ]
        i=max(range(len(arms)), key=lambda j: ucb[j]); return i, arms[i]
    def update(self, i, reward, **_):
        self.N[i]+=1; self.R[i]+=reward

# ---------- Thompson (Bernoulli) ----------
class ThompsonBernoulli(BaseOpt):
    def __init__(self):
        self.alpha:List[float]=[]; self.beta:List[float]=[]
    def _init(self,n):
        if not self.alpha: self.alpha=[1.0]*n; self.beta=[1.0]*n
    def select(self, arms):
        import random
        self._init(len(arms))
        s=[random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(arms))]
        i=max(range(len(arms)), key=lambda j:s[j]); return i, arms[i]
    def update(self, i, reward, **_):
        # ממפה תגמול רציף ל-הצלחה/כשל
        if reward>=0.5: self.alpha[i]+=1
        else: self.beta[i]+=1

# ---------- LinUCB (Contextual) ----------
class LinUCB(BaseOpt):
    """
    הקשר X \in R^d לכל זרוע; שומר A, b לכל זרוע. reward ~ x^T theta + noise.
    """
    def __init__(self, d:int=8, alpha:float=0.5):
        self.alpha=alpha; self.A=[]; self.b=[]; self.d=d
    def _init(self, n:int):
        if not self.A:
            import numpy as np
            self.A=[np.eye(self.d) for _ in range(n)]
            self.b=[np.zeros((self.d,)) for _ in range(n)]
    def select(self, arms):
        import numpy as np
        self._init(len(arms))
        # נדרש context לכל זרוע: arms[i]["x"] = np.array(d,)
        U=[]
        for i,a in enumerate(arms):
            x = np.array(a.get("x",""), dtype=float).reshape(-1)
            if x.size != self.d: x = np.zeros((self.d,)); x[0]=1.0
            Ainv = np.linalg.inv(self.A[i])
            theta = Ainv @ self.b[i]
            mu = float(theta @ x)
            var = float(math.sqrt(x @ Ainv @ x))
            U.append(mu + self.alpha*var)
        i = int(max(range(len(arms)), key=lambda j: U[j]))
        return i, arms[i]
    def update(self, i, reward, *, context=None):
        import numpy as np
        x = np.array((context or []), dtype=float).reshape(-1)
        if x.size != self.d: x = np.zeros((self.d,)); x[0]=1.0
        self.A[i] += np.outer(x,x)
        self.b[i] += reward * x

# ---------- Non-stationary wrapper (decay) ----------
class DecayWrapper(BaseOpt):
    def __init__(self, base: BaseOpt, decay: float = 0.97):
        self.base = base; self.decay = decay
    def select(self, arms): return self.base.select(arms)
    def update(self, i, reward, **kw):
        # decay על כל המדדים של הבייס (אם יש)
        for attr in ("R","N","alpha","beta"):
            if hasattr(self.base, attr):
                v=getattr(self.base, attr)
                if isinstance(v, list):
                    for j in range(len(v)): v[j]*=self.decay
        return self.base.update(i, reward, **kw)

# ---------- Dueling (RUCB מינימלי) ----------
class DuelingRUCB(BaseOpt):
    """שומר מטריצת ניצחונות; בכל בחירה מציע pair, העדפה → עדכון."""
    def __init__(self, n_max: int = 64):
        self.W:List[List[int]]=[]; self.C:List[List[int]]=[]; self.n=0; self.n_max=n_max
    def _ensure(self, n:int):
        if n>self.n:
            for _ in range(n-self.n):
                self.W.append([0]*n); self.C.append([0]*n)
            for i in range(self.n):
                self.W[i]+= [0]*(n-self.n); self.C[i]+= [0]*(n-self.n)
            self.n=n
    def select(self, arms):
        n=len(arms); self._ensure(n)
        # בחר שתי זרועות אקראיות (פשוט); פרקטית מחליפים ל-RUCB מלא
        import random
        i,j = random.sample(range(n), 2)
        return (i, {"duel_with": j, **arms[i]})
    def update(self, i, reward, *, duel_with=None):
        # reward>0.5 ⇒ i ניצח את j
        if duel_with is None: return
        self.C[i][duel_with]+=1; self.C[duel_with][i]+=1
        if reward>=0.5: self.W[i][duel_with]+=1
        else:           self.W[duel_with][i]+=1

# ---------- Pareto (רב-יעדי) ----------
def pareto_front(points: List[Dict[str,float]], keys: List[str]) -> List[int]:
    """
    מחזיר אינדקסים של חזית Pareto. מינימיזציה למפתחות עם prefix 'min_' ומקסימיזציה לאחרים.
    """
    idx=[]
    for i,pi in enumerate(points):
        dominated=False
        for j,pj in enumerate(points):
            if i==j: continue
            better_or_eq=True; strictly_better=False
            for k in keys:
                a=pi[k]; b=pj[k]
                if k.startswith("min_"):
                    if b>a: better_or_eq=False
                    if b<a: strictly_better=True
                else:
                    if b<a: better_or_eq=False
                    if b>a: strictly_better=True
            if better_or_eq and strictly_better:
                dominated=True; break
        if not dominated: idx.append(i)
    return idx

# ---------- AutoOpt (בחירה אוטומטית של אלגוריתם) ----------
class AutoOpt(BaseOpt):
    """
    בוחר אלגוריתם לפי הבעיה:
    - אם יש הקשר (context d>0) → LinUCB (עם Decay אם nonstationary)
    - אם success/fail → ThompsonBernoulli
    - אחרת → UCB1 (עם Decay אם nonstationary)
    אפשר לקבוע strategy=... ידנית.
    """
    def __init__(self, strategy: Optional[str]=None, d:int=8, alpha:float=0.5, nonstationary: bool=True):
        self.strategy=strategy; self.nonstat=nonstationary
        self.ucb  = UCB1()
        self.eg   = EpsGreedy(0.1)
        self.th   = ThompsonBernoulli()
        self.lin  = LinUCB(d=d, alpha=alpha)
        self._active: BaseOpt = self.ucb
    def _pick(self, arms: List[Dict[str,Any]]):
        strat = self.strategy
        if strat is None:
            has_ctx = all(isinstance(a.get("x"), (list,tuple)) for a in arms)
            if has_ctx: strat="linucb"
            elif any(a.get("binary") for a in arms): strat="thompson"
            else: strat="ucb1"
        if strat=="linucb": base=self.lin
        elif strat=="thompson": base=self.th
        elif strat=="egreedy": base=self.eg
        else: base=self.ucb
        self._active = DecayWrapper(base) if self.nonstat else base
    def select(self, arms):
        self._pick(arms); return self._active.select(arms)
    def update(self, i, reward, **kw):
        return self._active.update(i, reward, **kw)

# ---------- GP-BO (אופציונלי, אם sklearn מותקן) ----------
class GPBO:
    """
    Black-box optimization minimal: דורש scikit-learn. משתמש ב-GaussianProcessRegressor עם EI.
    """
    def __init__(self):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            self._ok=True
            self._GPR = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
        except Exception:
            self._ok=False
        self.X=[]; self.y=[]
    def suggest(self, bounds: List[Tuple[float,float]], n_suggestions:int=1) -> List[List[float]]:
        if not self._ok or not self.X:
            import random
            return [[ random.uniform(a,b) for (a,b) in bounds ] for _ in range(n_suggestions)]
        import numpy as np
        # EI גס
        Xs=[]
        for _ in range(64):
            x=[ random.uniform(a,b) for (a,b) in bounds ]; Xs.append(x)
        Xs=np.array(Xs)
        self._GPR.fit(self.X, self.y)
        mu, sigma = self._GPR.predict(Xs, return_std=True)
        ybest=max(self.y) if self.y else 0.0
        from math import erf, sqrt, exp
        def ei(m,s):
            if s<=1e-9: return 0.0
            z=(m-ybest)/(s+1e-9)
            Phi=0.5*(1+erf(z/math.sqrt(2))); phi=(1/math.sqrt(2*math.pi))*math.exp(-0.5*z*z)
            return (m-ybest)*Phi + s*phi
        EI=[ ei(m,s) for (m,s) in zip(mu,sigma) ]
        i=int(max(range(len(EI)), key=lambda j: EI[j]))
        return [Xs[i].tolist()]
    def observe(self, x: List[float], reward: float):
        import numpy as np
        self.X.append(x); self.y.append(reward)
