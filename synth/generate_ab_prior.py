# imu_repo/synth/generate_ab_prior.py
from __future__ import annotations
from typing import Dict, Any, List
import json

def generate_variants_with_prior(spec: Dict[str,Any], baseline: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    מייצר וריאציות בדמות ה-Baseline:
    - אם ה-Baseline מהיר/יציב (Φ טוב) → מייצרים וריאציה A "דומה" (#FAST) ועוד אלטרנטיבה B (#SLOW) לבקרת A/B.
    - אם baseline לא יציב בעתיד אפשר להחליף היגיון כאן (למשל לחקור וריאציות נוספות).
    שתי הווריאציות עומדות בתאימות פונקציונלית (מחזירות goal).
    """
    goal = json.dumps(spec["goal"], ensure_ascii=False)
    base_label = str(baseline.get("label","A")).upper()

    # "A_like": משמרים דפוס מהיר (בכוונה #FAST) כדי לתת prior שמזרז התכנסות.
    code_a_like = f"""#FAST
def helper_like():
    return 1  # שמירת דפוס פשוט ומהיר

def main():
    _ = helper_like()
    return {goal}
"""
    # B איטית כדי לשמר A/B (גם במצב שיש prior חזק)
    code_b = f"""#SLOW
def helper_heavy():
    acc = 0
    for i in range(20000):
        acc += i
    return acc

def main():
    _ = helper_heavy()
    return {goal}
"""
    return [
        {"label": f"{base_label}", "language":"python", "code":code_a_like},
        {"label": "B", "language":"python", "code":code_b},
    ]