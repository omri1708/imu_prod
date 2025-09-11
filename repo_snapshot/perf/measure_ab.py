# imu_repo/perf/measure_ab.py
from __future__ import annotations
from typing import Dict, Any

def measure_perf_for_code(code: str, language: str) -> Dict[str,Any]:
    """
    מדידה דטרמיניסטית ללא IO חיצוני:
    - p95_ms: פונקציה באורך הקוד ובדגלים (#SLOW, #FAST)
    - error_rate: אם מופיע '#ERROR' או אם אין 'return' בקוד (מייצג כשל ריצה/תוצאה)
    - cost_units: חישוב גס לפי אורך/מורכבות
    """
    base = max(1, len(code)//400)  # בערך ms
    p95 = float(base)

    if "#SLOW" in code:    p95 *= 10.0
    if "#FAST" in code:    p95 *= 0.5
    if "while True" in code or "time.sleep" in code:
        p95 *= 3.0

    error = 0.0
    if "#ERROR" in code or "raise " in code or "return" not in code:
        error = 1.0

    cost = float(len(code)/100.0)
    return {"p95_ms": p95, "error_rate": error, "cost_units": cost}