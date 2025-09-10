from __future__ import annotations
import re

_SYS_RE  = re.compile(r"(?is)\b(BEGIN|END)\s+SYSTEM(?:\s+OVERRIDE)?\b.*?")
_POLY_RE = re.compile(r"(?is)policy\s*:\s*override\s*=\s*true")

def sanitize(text: str) -> str:
    t = re.sub(_SYS_RE,  " ", text or "")
    t = re.sub(_POLY_RE, "policy: override=false", t)
    return t.strip()
