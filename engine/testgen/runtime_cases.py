from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, Any, List

OUT_DIR = Path("imu_repo/tests/generated/runtime_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _gauss_int(mu: float, sigma: float) -> int:
    return max(0, int(random.gauss(mu, sigma)))

def _mk_case_filter(mu: int, sigma: int, n: int) -> Dict[str, Any]:
    # סף פילטר סביב mu±sigma; דגימות סביב/מתחת/מעל
    thr = _gauss_int(mu, sigma)
    rows: List[Dict[str,Any]] = []
    for i in range(n):
        v = _gauss_int(mu, sigma)
        rows.append({"order_id": f"o{i:03d}", "amount": v})
    return {
        "spec": {
            "path": "page.components[0]",
            "binding_url": "https://api.example.com/orders",
            "columns": [
                {"name": "order_id", "type":"string", "required": True},
                {"name": "amount",   "type":"number", "required": True},
            ],
            "filters": {"amount": {"op": ">=", "value": thr}},
            "sort": None
        },
        "rows": rows,
        "policy": {
            "runtime_check_enabled": True,
            "allow_remove_filter_if_blocked": True,
            "auto_remediation": {"enabled": True, "apply_levels":["conservative"], "max_rounds": 1}
        }
    }

def _mk_case_required(missing_prob: float, n: int) -> Dict[str, Any]:
    rows = []
    for i in range(n):
        has_amount = random.random() > missing_prob
        row = {"order_id": f"o{i:03d}"}
        if has_amount:
            row["amount"] = random.randint(1, 200)
        rows.append(row)
    return {
        "spec": {
            "path": "page.components[0]",
            "binding_url": "https://api.example.com/orders",
            "columns": [
                {"name":"order_id","type":"string","required": True},
                {"name":"amount",  "type":"number","required": True},
            ],
            "filters": None, "sort": None
        },
        "rows": rows,
        "policy": {
            "runtime_check_enabled": True,
            "allow_relax_required_if_missing": True,
            "auto_remediation": {"enabled": True, "apply_levels":["conservative"], "max_rounds": 1}
        }
    }

def _mk_case_drift(n1: int, n2: int) -> Dict[str, Any]:
    r1 = [{"id": f"u{i:03d}"} for i in range(n1)]
    r2 = [{"id": f"u{i:03d}"} for i in range(n2)]
    return {
        "spec": {
            "path": "page.components[0]",
            "binding_url": "https://api.example.com/users",
            "columns": [{"name":"id","type":"string","required": True}],
            "filters": None, "sort": None
        },
        "rows_v1": r1,
        "rows_v2": r2,
        "policy": {
            "runtime_check_enabled": True,
            "block_on_drift": True,
            "allow_update_prev_hash_on_schema_ok": True,
            "runtime_prev_hash_map": {},
            "auto_remediation": {"enabled": True, "apply_levels":["conservative"], "max_rounds": 1},
            "runtime_state_dir": "runs/runtime_state"
        }
    }

def main(seed: int = 1337) -> None:
    random.seed(seed)
    # גאוס סביב 100 עם סטייה 25, וקצוות אקסטרים
    cases = []
    for _ in range(8):
        cases.append(_mk_case_filter(mu=100, sigma=25, n=30))
    # קצוות: סף גבוה מאוד / נמוך מאוד
    cases.append(_mk_case_filter(mu=5, sigma=2, n=20))
    cases.append(_mk_case_filter(mu=200, sigma=5, n=20))
    # required – 30% חסר
    cases.append(_mk_case_required(missing_prob=0.3, n=30))
    # drift – בונים שינוי בכמות הרשומות
    cases.append(_mk_case_drift(n1=20, n2=27))

    for i, c in enumerate(cases):
        (OUT_DIR / f"runtime_case_{i:02d}.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
