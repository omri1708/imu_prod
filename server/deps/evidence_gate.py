# server/deps/evidence_gate.py
from fastapi import Request
ENFORCED_PREFIXES = ("/respond", "/chat", "/orchestrate", "/build", "/program")

async def require_citations_or_silence(request: Request) -> None:
    # מסמן שהמסלול הזה מחויב בראיות
    request.state.require_evidence = any(request.url.path.startswith(p) for p in ENFORCED_PREFIXES)

def has_citations(data: dict) -> bool:
    ev = data.get("evidence") if isinstance(data, dict) else None
    if not isinstance(ev, dict):
        return False
    cits = ev.get("citations")
    return isinstance(cits, list) and len(cits) > 0
