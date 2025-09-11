from __future__ import annotations
from typing import Dict, Any, List

CHECKS_FOR_CATEGORY = {
  "web.frontend": ["exe:node", "exe:npm"],
  "mobile.ios":   ["exe:xcodebuild"],
  "mobile.android":["exe:sdkmanager","exe:gradle"],
  "realtime":     ["exe:ffmpeg"],
  "gpu.cuda":     ["exe:nvcc","py:torch"],
  "k8s":          ["exe:kubectl","exe:helm"],
  "db.sql":       ["exe:sqlite3"],
}

def required_approvals(spec: Dict[str, Any]) -> Dict[str, Any]:
    req = {"tools": [], "licenses": [], "secrets": [], "notes": []}
    for t in (spec.get("tools") or []):
        cat = (t or {}).get("category")
        if cat in CHECKS_FOR_CATEGORY:
            req["tools"] += CHECKS_FOR_CATEGORY[cat]
    blob = str(spec).lower()
    if "ios" in blob or "xcode" in blob:
        req["notes"].append("Requires macOS + Xcode + provisioning (GUI/EULA).")
    if "cuda" in blob or "gpu" in blob:
        req["notes"].append("Requires NVIDIA GPU + CUDA toolkit.")
    if "stripe" in blob:
        req["secrets"].append("STRIPE_KEY")
    if "openai" in blob:
        req["secrets"].append("OPENAI_API_KEY")
    req["tools"] = sorted(set(req["tools"]))
    req["secrets"] = sorted(set(req["secrets"]))
    return req
