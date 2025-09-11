# imu_repo/ui_dsl/versioning.py
from __future__ import annotations
import json, hashlib, time
from typing import Dict, Any, List
from cas.store import put_json

def app_version_manifest(*, ui_spec: Dict[str,Any], assets: List[Dict[str,Any]], policy: Dict[str,Any]) -> Dict[str,Any]:
    payload = {
        "kind":"ui_app_version",
        "ui_spec": ui_spec,            # מובטח דטרמיניסטי במסגור JSON
        "assets": assets,
        "policy_fp": hashlib.sha256(json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "ts": time.time()
    }
    # חישוב hash לקביעה חד-חד ערכיות
    b = json.dumps(payload, ensure_ascii=False, separators=(",",":")).encode("utf-8")
    sha = hashlib.sha256(b).hexdigest()
    payload["sha256"] = sha
    saved = put_json(payload)
    return {"ok": True, "sha256": sha, "manifest_sha256": saved["sha256"], "manifest": payload}