# imu_repo/engine/official_gate.py
from __future__ import annotations
from typing import Dict, Any, List
from grounded.claims import current
from verifiers.official_verify import verify_official_payload
from verifiers.official_registry import get_official

def run_official_checks(cfg: Dict[str,Any]) -> None:
    """
    סורק את הראיות שבקונטקסט; עבור כל ראיה שכוללת payload עם official{source_id,signature}
    מאמת חתימה ומוסיף ראיית 'official_verified' (עם trust לפי אמון המקור).
    פועל אידמפוטנטי (לא מוסיף כפילות).
    """
    evs = current().snapshot()
    already = {(e.get("kind"), e.get("payload", {}).get("ref_sha256")) for e in evs if e.get("kind") == "official_verified"}
    # נעבור על כל הראיות שקיימות
    for ev in evs:
        payload = ev.get("payload", {})
        off = payload.get("official")
        if not isinstance(off, dict):
            continue
        ok, why = verify_official_payload(payload)
        ref_sha = ev.get("sha256")
        if ("official_verified", ref_sha) in already:
            continue
        if ok:
            src_id = str(off.get("source_id"))
            rec = get_official(src_id)
            src_trust = float(rec.get("trust", 0.9)) if rec else 0.7
            current().add_evidence("official_verified", {
                "source_url": f"official://{src_id}",
                "trust": min(0.995, src_trust),
                "ttl_s": float(ev.get("ttl_s", 600.0)),
                "payload": {"ref_sha256": ref_sha, "source_id": src_id, "result": "verified"}
            })
        else:
            # גם כשל הוא ראיה – שקיפות מלאה
            current().add_evidence("official_verification_failed", {
                "source_url": "local://official_gate",
                "trust": 0.9,
                "ttl_s": float(ev.get("ttl_s", 600.0)),
                "payload": {"ref_sha256": ref_sha, "reason": why}
            })