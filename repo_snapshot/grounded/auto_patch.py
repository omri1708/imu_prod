# imu_repo/grounded/auto_patch.py
from __future__ import annotations
from typing import Dict, Any, List
import json, time, os
from grounded.evidence_policy import policy_singleton as EvidencePolicy
from grounded.contradiction_policy import policy_singleton as ContraPolicy, MetricRule
from grounded.provenance import ProvenanceStore

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_audit_line(audit_path: str, event: Dict[str,Any]) -> None:
    _ensure_dir(os.path.dirname(audit_path))
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **event}, ensure_ascii=False) + "\n")

def auto_patch_from_resolution(build_dir: str,
                               pv: ProvenanceStore,
                               resolution: Dict[str,Any],
                               *,
                               tighten_trust_by_key: Dict[str, float] = None,
                               min_consistency_score: float = None) -> Dict[str,Any]:
    """
    מחיל תיקונים שמרניים:
      - מחזק ספי min_trust לראיות שנפלו (dropped).
      - מחמיר tol למדדים קריטיים אם נצפתה סטייה גבוהה.
      - רושם evidence 'resolution_proof' ל-provenance.
      - רושם שורת audit.
    """
    tighten_trust_by_key = tighten_trust_by_key or {}
    proof = resolution.get("proof", {})
    dropped = resolution.get("dropped", {})

    # 1) הידוק אמון לראיות שנפסלו
    raised: Dict[str, float] = {}
    for m, arr in dropped.items():
        # עבור מטריקה, העלה min_trust לכל evidence key שהודר — לפחות לערך cut
        for (key, _val, trust) in arr:
            # נעלה ל-max(cut, existing, explicit)
            target = max(float(proof.get("trust_cut", 0.75)),
                         float(EvidencePolicy.min_trust_by_key.get(key, 0.0)),
                         float(tighten_trust_by_key.get(key, 0.0)))
            if target > EvidencePolicy.min_trust_by_key.get(key, 0.0):
                EvidencePolicy.set_min_trust(key, target)
                raised[key] = target

    # 2) החמרת כללי סתירה אם צריך
    if isinstance(min_consistency_score, (int, float)) and min_consistency_score > ContraPolicy.min_consistency_score:
        ContraPolicy.set_min_score(float(min_consistency_score))

    # 3) הוספת evidence חדשה עם הוכחת רזולוציה
    rec = pv.put("resolution_proof", {
        "effective": resolution.get("effective", {}),
        "used": resolution.get("used", {}),
        "dropped": resolution.get("dropped", {}),
        "proof": proof
    }, source_url="internal.test://resolution", trust=0.99)

    # 4) Audit
    audit_path = os.path.join(build_dir, "audit", "autopatch.jsonl")
    write_audit_line(audit_path, {
        "event": "auto_patch",
        "raised_min_trust": raised,
        "min_consistency_score": ContraPolicy.min_consistency_score,
        "res_evidence_id": rec["_id"]
    })
    return {"raised_min_trust": raised, "res_id": rec["_id"], "min_consistency_score": ContraPolicy.min_consistency_score}