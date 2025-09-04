# imu_repo/engine/fallbacks.py
from __future__ import annotations
import time
from typing import Dict, Any
from grounded.claims import current

# imu_repo/engine/fallbacks.py
from __future__ import annotations
from typing import Dict, Any

def safe_text_fallback(*, reason: str, details: Dict[str,Any] | None = None) -> str:
    d = details or {}
    parts = [f"[FALLBACK] {reason}"]
    if d:
        parts.append(str(d))
    return " | ".join(parts)


