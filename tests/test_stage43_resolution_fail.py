# imu_repo/tests/test_stage43_resolution_fail.py
from __future__ import annotations
import tempfile, shutil
from grounded.provenance import ProvenanceStore
from grounded.contradiction_resolution import resolve_contradictions

def run():
    tmp = tempfile.mkdtemp(prefix="imu_pv_")
    try:
        pv = ProvenanceStore(tmp)
        # שלוש ראיות בינוניות אמון, סותרות חזק — אחרי trust_cut=0.8 לא יישאר כלום/יישארו סתירות
        pv.put("m1", {"p95_ms": 1500.0}, source_url="news.example://a", trust=0.79)
        pv.put("m2", {"p95_ms": 100.0},  source_url="wiki.example://b", trust=0.78)
        pv.put("m3", {"p95_ms": 800.0},  source_url="user.example://c", trust=0.60)

        res = resolve_contradictions(pv, ["m1","m2","m3"], trust_cut=0.8)
        ok = (not res.ok)  # לא מצליח לפתור → זה המצב המצופה
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__=="__main__":
    raise SystemExit(run())