# imu_repo/engine/convergence.py
from __future__ import annotations
from typing import List, Tuple

def moving_average(xs: List[float], window: int = 20) -> List[float]:
    if window <= 0:
        raise ValueError("window>0 required")
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for x in xs:
        q.append(float(x)); s += float(x)
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def has_converged(xs: List[float], window: int = 20, rel_tol: float = 0.02, strict_tail: int = 5) -> bool:
    """
    התכנסות אמפירית: ממוצע נע אחרון נמוך ב־rel_tol לפחות מהממוצע בתחילת החלון,
    וש־strict_tail הערכים האחרונים אינם עולים (non-increasing).
    """
    if len(xs) < max(window, strict_tail):
        return False
    ma = moving_average(xs, window)
    if len(ma) < window:
        return False
    head = ma[-window]
    tail = ma[-1]
    if head <= 0:
        return False
    improved = (tail <= head * (1.0 - rel_tol))
    # non-increasing tail
    ni = all(ma[-i] <= ma[-i-1] for i in range(1, min(strict_tail, len(ma))))
    return bool(improved and ni)

def regression_guard(phi_new: float, phi_baseline: float, promote_margin: float = 0.01) -> bool:
    """
    מאשר קידום רק אם יש שיפור יחסי של לפחות promote_margin (ברירת מחדל: ≥1% שיפור).
    """
    if phi_baseline <= 0:
        return True
    return (phi_new <= phi_baseline * (1.0 - promote_margin) + 1e-9)