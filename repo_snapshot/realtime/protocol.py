# imu_repo/realtime/protocol.py
from __future__ import annotations
import json
from typing import Any, Dict, Tuple

class ProtocolError(Exception):
    pass

# מעטפת קנונית: כל הודעה היא JSON עם שני שדות:
#  - "op": שם פעולה לוגי (string)
#  - "bundle": עצם הנתונים (לרוב {"text":..., "claims":[...]} או פקודת בקרת-ערוץ)
# זה מאפשר הרחבה דטרמיניסטית, וחיבור StrictSink לפני/אחרי רשת.

def pack(op: str, bundle: Dict[str, Any]) -> bytes:
    if not isinstance(op, str) or not op:
        raise ProtocolError("op must be non-empty string")
    if not isinstance(bundle, dict):
        raise ProtocolError("bundle must be dict")
    doc = {"op": op, "bundle": bundle}
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def unpack(b: bytes) -> Tuple[str, Dict[str, Any]]:
    try:
        doc = json.loads(b.decode("utf-8"))
    except Exception as e:
        raise ProtocolError(f"invalid json: {e}")
    if not isinstance(doc, dict) or "op" not in doc or "bundle" not in doc:
        raise ProtocolError("message must contain 'op' and 'bundle'")
    op = doc["op"]
    bundle = doc["bundle"]
    if not isinstance(op, str) or not isinstance(bundle, dict):
        raise ProtocolError("bad field types")
    return op, bundle

# סכימה מינימלית ל-bundle מסוג "grounded_text":
# {
#   "text": "string",
#   "claims": [{"type": "...", "text": "...", "evidence": [...]}],
#   "meta": {...}   # אופציונלי
# }
def require_grounded_bundle(bundle: Dict[str, Any]) -> None:
    if "text" not in bundle or "claims" not in bundle:
        raise ProtocolError("grounded bundle must include text and claims")
    if not isinstance(bundle["text"], str):
        raise ProtocolError("text must be string")
    if not isinstance(bundle["claims"], list) or not bundle["claims"]:
        raise ProtocolError("claims must be non-empty list")
    for i, c in enumerate(bundle["claims"]):
        if not isinstance(c, dict):
            raise ProtocolError(f"claim[{i}] must be object")
        if "type" not in c or "text" not in c:
            raise ProtocolError(f"claim[{i}] missing type/text")
        ev = c.get("evidence", [])
        if not isinstance(ev, list) or not ev:
            raise ProtocolError(f"claim[{i}] must include non-empty evidence[]")