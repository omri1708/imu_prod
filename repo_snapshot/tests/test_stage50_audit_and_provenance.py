# imu_repo/tests/test_stage50_audit_and_provenance.py
from __future__ import annotations
import os, json
from audit.cas import put_json, get_path
from audit.ledger import append, verify_chain
from audit.provenance_store import record_evidence

def run():
    a = put_json({"hello":"world"}, meta={"kind":"sample"})
    b = record_evidence("unit", {"x":1}, actor="tester", obj="thing:1", tags=["t"])
    ok1 = os.path.exists(get_path(a["sha256"]))
    ok2 = verify_chain()
    append({"actor":"tester","action":"noop","object":"x"})  # עוד חוליה
    ok3 = verify_chain()
    print("OK" if (ok1 and ok2 and ok3) else "FAIL")
    return 0 if (ok1 and ok2 and ok3) else 1

if __name__=="__main__":
    raise SystemExit(run())