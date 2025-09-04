# imu_repo/tests/test_stage42_consistency_fail_local.py
from __future__ import annotations
import os, json, time, tempfile, shutil
from grounded.provenance import ProvenanceStore
from grounded.consistency import analyze_consistency
from grounded.source_policy import policy_singleton as Policy

def run():
    tmp = tempfile.mkdtemp(prefix="imu_pv_")
    try:
        pv = ProvenanceStore(tmp)
        # מכניסים שתי ראיות על אותו מדד perf.p95_ms עם ערכים שונים מאוד
        pv.put("perf_summary", {"p95_ms": 180.0}, source_url="internal.test://evidence", trust=0.99)
        pv.put("ext_perf", {"p95_ms": 2000.0}, source_url="user.example://report", trust=0.4)
        # מוסיפים גם ui_accessibility עקבי כדי שלא יפריע
        pv.put("ui_accessibility", {"score": 90}, source_url="internal.test://evidence", trust=0.95)

        res = analyze_consistency(pv, ["perf_summary","ext_perf","ui_accessibility"])
        ok = (not res["ok"]) and any(c["metric"]=="perf.p95_ms" for c in res["contradictions"])
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__=="__main__":
    raise SystemExit(run())