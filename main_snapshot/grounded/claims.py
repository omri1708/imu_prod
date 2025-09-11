# imu_repo/grounded/claims.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Optional
from grounded.provenance import STORE
from grounded.provenance_confidence import normalize_and_sign
from engine.config import load_config
from engine.policy_ctx import get_user

os.makedirs(STORE, exist_ok=True)

class _ClaimCtx:
    def __init__(self) -> None:
        self._evidences: List[Dict[str,Any]] = []

    def reset(self) -> None:
        self._evidences.clear()

    def add_evidence(self, kind: str, info: Dict[str,Any]) -> None:
        cfg = load_config()
        secret = str(cfg.get("evidence", {}).get("signing_secret", "imu_default_secret"))
        ev = normalize_and_sign(kind, info, signing_secret=secret)
        # תיוג משתמש
        uid = get_user() or "anon"
        ev["user_id"] = uid
        self._evidences.append(ev)
        # שמירת קובץ — מבודד בשם (כולל user)
        ts = ev.get("ts", time.time())
        safe_kind = "".join(ch if ch.isalnum() else "_" for ch in kind)
        fn = os.path.join(STORE, f"{int(ts*1000)}__{uid}__{safe_kind}.json")
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False)
        except Exception:
            pass

    def snapshot(self) -> List[Dict[str,Any]]:
        return list(self._evidences)

# singleton
_CTX = _ClaimCtx()

def current() -> _ClaimCtx:
    return _CTX