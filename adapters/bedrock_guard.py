from __future__ import annotations
import os
from typing import Dict, Any

def bedrock_guard(prompt: str, *, allow_live: bool) -> Dict[str,Any]:
    if not allow_live or not os.environ.get("AWS_REGION"):
        return {"ok": True, "mode": "dryrun", "request": {"text": prompt}}
    return {"ok": True, "mode": "live", "request": {"text": prompt}}
