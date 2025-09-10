from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple

class LocalizationResult(Exception):
    pass

def bisect_steps(steps: List[Callable[[], Dict[str,Any]]]) -> Tuple[int, Dict[str,Any]]:
    """
    מחזיר את אינדקס ה'שלב השובר' = השלב הראשון שמחזיר ok=False (או זורק).
    רץ בלוג N (bisection) כדי לאתר מקור שורשי מהר.
    """
    lo, hi = 0, len(steps)-1
    last_ok: Dict[str,Any] = {}
    while lo <= hi:
        mid = (lo + hi)//2
        ok = True; out = {}
        for i in range(0, mid+1):
            out = steps[i]()
            if not bool(out.get("ok", True)):
                ok = False; break
        if ok:
            last_ok = out
            lo = mid + 1
        else:
            hi = mid - 1
    if lo >= len(steps):
        raise LocalizationResult({"last_ok": last_ok})
    return lo, {"last_ok": last_ok}   # ← זהו "השובר"
