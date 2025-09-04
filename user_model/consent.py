# imu_repo/user_model/consent.py
from __future__ import annotations
from typing import Dict, Any
import os, json, time
from user_model.identity import user_dir
from privacy.storage import save_json_encrypted, load_json_encrypted

CONSENT_FN = "consent.json"

from __future__ import annotations
from typing import Dict, Any


DEFAULT = {
    "analytics": False,
    "personalization": True,
    "cross_session_learning": True,
    "share_evidence_external": False,
}


def set_consent(user_id: str, **flags) -> Dict[str, Any]:
    m = dict(DEFAULT)
    m.update({k: bool(v) for k,v in flags.items()})
    save_json_encrypted(user_id, "consent", m, ttl_s=None)
    return m


def get_consent(user_id: str) -> Dict[str, Any]:
    m = load_json_encrypted(user_id, "consent")
    return dict(DEFAULT) if m is None else dict(m)

def require(user_id: str, *, personalization: bool | None = None, cross_session: bool | None = None) -> None:
    m = get_consent(user_id)
    if personalization is True and not m.get("personalization", False):
        raise PermissionError("personalization_not_allowed")
    if cross_session is True and not m.get("cross_session_learning", False):
        raise PermissionError("cross_session_learning_not_allowed")


def _path(user_key: str) -> str:
    return os.path.join(user_dir(user_key), CONSENT_FN)


def revoke(user_key: str, purpose: str) -> None:
    p = _path(user_key)
    try:
        data = json.load(open(p,"r",encoding="utf-8"))
    except Exception:
        data = {}
    if purpose in data:
        data[purpose]["granted"] = False
        data[purpose]["ts"] = time.time()
    json.dump(data, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def check(user_key: str, purpose: str) -> Dict[str,Any]:
    p = _path(user_key)
    try:
        data = json.load(open(p,"r",encoding="utf-8"))
    except Exception:
        return {"ok": False, "reason": "no_record"}
    rec = data.get(purpose)
    if not rec: return {"ok": False, "reason": "no_record"}
    if not rec.get("granted", False): return {"ok": False, "reason": "revoked"}
    alive = (time.time() <= rec["ts"] + rec["ttl_s"])
    return {"ok": alive, "reason": None if alive else "expired", "record": rec}