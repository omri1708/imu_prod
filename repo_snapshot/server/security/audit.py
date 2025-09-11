# server/security/audit.py
from pathlib import Path
import json, time

AUDIT_FILE = Path("./var/audit.log")
AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

def audit_log(event: str, data: dict):
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "event": event, "data": data}, ensure_ascii=False) + "\n")