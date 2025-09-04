# imu_repo/engine/gates/user_conflict_gate.py
from __future__ import annotations
from typing import Dict, Any, List

class UserConflictGate:
    """
    בודק 'אמביגואיות/סתירה' בעוצמה גבוהה בפרופיל T1/T2:
      - keys: רשימת מפתחות 'קריטיים' לבדיקה (אם ריק, בודק את כולם).
      - max_ambiguity: רף אמביגואיות מותרת (מרחק מ-0.5; נמוך=לא החלטי).
      - min_strength: משקל מינימלי (n) שנדרש כדי להחליט (נמוך מדי => אמביגואי).
    """
    def __init__(self, keys: List[str] | None=None, max_ambiguity: float=0.2, min_strength: float=0.5):
        self.keys = keys or []
        self.max_ambiguity = float(max_ambiguity)
        self.min_strength  = float(min_strength)

    def check(self, profile: Dict[str,Any]) -> Dict[str,Any]:
        prefs = profile.get("pref", {})
        beliefs = profile.get("beliefs", {})
        strength = profile.get("strength", {})
        def amb(mu: float) -> float:  # אמביגואיות = כמה קרוב ל-0.5
            return abs(0.5 - float(mu))

        crit = self.keys or sorted(set(list(prefs.keys()) + list(beliefs.keys())))
        offenders=[]
        for k in crit:
            mu = prefs.get(k, beliefs.get(k))
            if mu is None: 
                offenders.append((k, "missing"))
                continue
            s  = float(strength.get(k, 0.0))
            if s < self.min_strength:
                offenders.append((k, f"weak:{s:.3f}"))
                continue
            if amb(mu) < self.max_ambiguity:
                offenders.append((k, f"ambiguous:mu={mu:.3f}"))
        ok = (len(offenders)==0)
        return {"ok": ok, "offenders": offenders, "checked": crit}