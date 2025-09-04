# imu_repo/synth/generate_ab.py
from __future__ import annotations
from typing import Dict, Any, List
import json

def generate_variants(spec: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    מייצר שתי וריאציות קוד דטרמיניסטיות:
    - A: "מהיר" (#FAST)
    - B: "איטי" (#SLOW)
    שתיהן מחזירות את ה-goal כדי לשמר תאימות פונקציונלית.
    """
    goal = json.dumps(spec["goal"], ensure_ascii=False)
    code_a = f"""#FAST
def main():
    # וריאציה A — פשוטה
    return {goal}
"""
    code_b = f"""#SLOW
def helper():
    # סימולציית איטיות לוגית
    acc = 0
    for i in range(10000):  # עבודה "כבדה"
        acc += i
    return acc

def main():
    x = helper()
    return {goal}
"""
    return [
        {"label":"A","language":"python","code":code_a},
        {"label":"B","language":"python","code":code_b},
    ]