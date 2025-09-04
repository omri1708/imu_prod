# imu_repo/security/fingerprint_report.py
from __future__ import annotations
import os, json, hashlib, time, urllib.request, urllib.error

OUTBOX = os.environ.get("IMU_FINGERPRINT_OUTBOX", "/mnt/data/imu_fingerprints_outbox")
ENDPOINT = os.environ.get("IMU_FINGERPRINT_URL", "")

def compute_fingerprint(doc: dict) -> dict:
    data = json.dumps(doc, sort_keys=True).encode("utf-8")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "ts": int(time.time()),
        "kind": doc.get("_type","manifest")
    }

def _post_json(url: str, doc: dict, timeout: float = 2.5) -> None:
    req = urllib.request.Request(url, data=json.dumps(doc).encode("utf-8"),
                                 headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        _ = r.read()

def _spool(doc: dict) -> str:
    os.makedirs(OUTBOX, exist_ok=True)
    path = os.path.join(OUTBOX, f"{int(time.time()*1000)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path

def report_fingerprint(doc: dict) -> dict:
    fp = compute_fingerprint(doc)
    payload = {"_type":"fingerprint","fp":fp,"doc":doc}
    if ENDPOINT:
        try:
            _post_json(ENDPOINT, payload)
            return {"ok": True, "mode":"http", "endpoint": ENDPOINT, "fp": fp}
        except Exception as e:
            path = _spool(payload)
            return {"ok": True, "mode":"outbox", "path": path, "reason": str(e), "fp": fp}
    else:
        path = _spool(payload)
        return {"ok": True, "mode":"outbox", "path": path, "fp": fp}