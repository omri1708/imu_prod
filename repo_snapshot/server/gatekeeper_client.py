# server/gatekeeper_client.py
# Helper קטן לקריאת Gatekeeper Evaluate דרך HTTP – ללא תלות חיצונית.
from __future__ import annotations
from typing import Dict, Any
import urllib.request, json

GATE_API = "http://127.0.0.1:8000"

def evaluate(gate: Dict[str,Any]) -> Dict[str,Any]:
    """
    gate: {"evidences":[...], "checks":{...}, "p95":{...}}
    → {"ok": bool, "reasons":[...]}
    """
    req = urllib.request.Request(
        GATE_API + "/gatekeeper/evaluate",
        method="POST",
        data=json.dumps(gate or {}).encode("utf-8"),
        headers={"Content-Type":"application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))