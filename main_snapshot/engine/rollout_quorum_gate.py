# imu_repo/engine/rollout_quorum_gate.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, Iterable, Callable
from engine.quorum_verify import quorum_verify
from engine.key_delegation import enforce_scope_for_kid, DelegationError

def _audit_dir() -> str:
    d = os.environ.get("IMU_AUDIT_DIR") or ".audit"
    os.makedirs(d, exist_ok=True)
    return d

def _append_audit(event: Dict[str,Any]) -> None:
    path = os.path.join(_audit_dir(), "rollout_gate.jsonl")
    event = {"ts": time.time(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def gate_release(bundle: Dict[str,Any], policy: Dict[str,Any], *, verifiers: Iterable[Callable[[Dict[str,Any],Dict[str,Any]], Dict[str,Any]]], k: int, expected_scope: str) -> Dict[str,Any]:
    """
    - בודק קודם שהמפתח החותם (key_id) מורשה ל-scope המבוקש (עפ"י שרשרת ההאצלות שבידי המאמתים).
      * כאן נדרוש שה-verifier הראשון יחזיר גם 'chain' אם הוא בנוי על KeychainManager — אחרת,
        ה-enforce_scope מתבצע בתוך המאמתים עצמם (ראה as_quorum_member_with_chain בהמשך), כך שהכשל יתועד ב-quorum.
    - מפעיל quorum k-of-n.
    - כותב תוצאת החלטה ללוג.
    """
    key_id = None
    try:
        sig = bundle.get("signature") or {}
        key_id = sig.get("key_id")
    except Exception:
        key_id = None

    # מריצים Quorum
    try:
        out = quorum_verify(bundle, policy, verifiers, k=k)
        _append_audit({"evt":"rollout_gate_pass","k":k,"oks":out.get("oks"),"total":out.get("total"),"key_id":key_id})
        return {"ok": True, **out}
    except Exception as e:
        _append_audit({"evt":"rollout_gate_fail","k":k,"err":str(e),"key_id":key_id})
        raise