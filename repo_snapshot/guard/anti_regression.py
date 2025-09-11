# imu_repo/guard/anti_regression.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time

HIST_PATH = "/mnt/data/imu_repo/history/kpi_history.jsonl"

class AntiRegressionResult(Dict[str,Any]): ...
class AntiRegressionError(Exception): ...

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _load_history() -> List[Dict[str,Any]]:
    if not os.path.exists(HIST_PATH): return []
    out=[]
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            try: out.append(json.loads(ln))
            except: pass
    return out

def _write_history(entry: Dict[str,Any]) -> None:
    _ensure_dir(os.path.dirname(HIST_PATH))
    with open(HIST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def check_and_record(*, service: str, kpi_score: float, p95_ms: float,
                     max_allowed_regression_pct: float = 7.5,
                     min_allowed_kpi: float = 70.0) -> AntiRegressionResult:
    """
    בודק מול הממוצע המשוקלל האחרון (או הערך האחרון אם אין מספיק נתונים).
    מחזיר:
      - ok: True/False
      - reason
      - baseline
    ובכל מקרה רושם את המדידה.
    """
    hist = _load_history()
    same = [h for h in hist if h.get("service")==service]
    if same:
        # בסיס פשטני: לוקחים KPI אחרון כ-baseline
        baseline = float(same[-1].get("kpi_score", 75.0))
    else:
        baseline = 75.0

    # תנאי סף
    if kpi_score < min_allowed_kpi:
        res = AntiRegressionResult(ok=False, reason="below_min_kpi", baseline=baseline)
    else:
        allowed = baseline * (1.0 - max_allowed_regression_pct/100.0)
        res = AntiRegressionResult(ok=(kpi_score >= allowed), reason=("ok" if kpi_score>=allowed else "regression"), baseline=baseline)

    # רישום
    _write_history({"ts": time.time(), "service": service, "kpi_score": kpi_score, "p95_ms": p95_ms})
    return res