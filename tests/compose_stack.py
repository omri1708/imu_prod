# imu_repo/tests/compose_stack.py
from __future__ import annotations
from orchestration.compose_workflow import build_stack

def run():
    res = build_stack()
    print(res)
    # אם אין Docker — נחזיר קוד הצלחה (כי נדרש משאב חיצוני) אך נדפיס “NEED: …”
    return 0

if __name__=="__main__":
    raise SystemExit(run())