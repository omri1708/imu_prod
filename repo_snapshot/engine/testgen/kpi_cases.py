from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Any

OUT_DIR = Path("imu_repo/tests/generated/kpi_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _mk_kpi_series(mu: float, sigma: float, n: int) -> List[Dict[str,Any]]:
    out = []
    for _ in range(n):
        v = max(1.0, random.gauss(mu, sigma))
        out.append({"ok": True, "latency_ms": round(v, 2)})
    return out

def main(seed: int = 42) -> None:
    random.seed(seed)
    # baseline סביב 80±10; candidate סביב 115±10
    base = _mk_kpi_series(80, 10, 30)
    cand = _mk_kpi_series(115, 10, 30)
    tmp = Path("imu_repo/tests/generated/kpi_cases")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "baseline.jsonl").write_text("\n".join(json.dumps(x) for x in base), encoding="utf-8")
    (tmp / "candidate.jsonl").write_text("\n".join(json.dumps(x) for x in cand), encoding="utf-8")

if __name__ == "__main__":
    main()
