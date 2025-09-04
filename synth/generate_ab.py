# imu_repo/synth/generate_ab_explore.py
from __future__ import annotations
from typing import Dict, Any, List
import json

def generate_variants_with_prior_and_explore(spec: Dict[str,Any], baseline: Dict[str,Any]) -> List[Dict[str,Any]]:
    goal = json.dumps(spec["goal"], ensure_ascii=False)
    base_label = str(baseline.get("label","A")).upper()

    code_a_like = f"""#FAST
def helper_like():
    return 1
def main():
    _ = helper_like()
    return {goal}
"""

    # ★ מוסיפים תג #EXPLORE כדי לאותת לסימולציה על סיכון יציבות קל (errors/trust)
    code_e_min = f"""#FAST #EXPLORE
def main():
    return {goal}
"""
    return [
        {"label": f"{base_label}", "language":"python", "code":code_a_like},
        {"label": "E", "language":"python", "code":code_e_min},
    ]