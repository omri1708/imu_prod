# imu_repo/engine/quorum_verify.py
from __future__ import annotations
from typing import Callable, Dict, Any, Iterable, List, Tuple

class QuorumError(Exception): ...

VerifierFn = Callable[[Dict[str,Any], Dict[str,Any]], Dict[str,Any]]
# חתימה: verifier(bundle, policy) -> {"ok":True} או {"ok":False,"reason":"..."}

def quorum_verify(bundle: Dict[str,Any], policy: Dict[str,Any], verifiers: Iterable[VerifierFn], *, k: int) -> Dict[str,Any]:
    """
    מריץ כמה מאמתים בלתי תלויים ודורש k הצלחות לפחות.
    """
    oks = 0
    reasons: List[str] = []
    total = 0
    for v in verifiers:
        total += 1
        try:
            out = v(bundle, policy)
            if out.get("ok"):
                oks += 1
            else:
                reasons.append(str(out.get("reason","failed")))
        except Exception as e:
            reasons.append(str(e))
    if oks >= k:
        return {"ok": True, "oks": oks, "total": total}
    raise QuorumError(f"quorum failed: oks={oks}/{total}, need k={k}; reasons={reasons}")