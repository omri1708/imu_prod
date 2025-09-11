# imu_repo/tools/imu_audit_dump.py
from __future__ import annotations
import json
from engine.audit_log import verify_chain, AUDIT_PATH

def main():
    res = verify_chain()
    print(json.dumps({"audit_path": AUDIT_PATH, **res}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())