# imu_repo/grounded/provenance_confidence.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, List, Optional
from grounded.provenance import STORE

STATE_DIR = "/mnt/data/imu_repo/.state"
SRC_DB = os.path.join(STATE_DIR, "sources.json")
os.makedirs(STATE_DIR, exist_ok=True)

_DEFAULTS = {
    # fallback ליוצרי־קבצים מקומיים (evidence עם source_url שמתחיל ב-local://)
    "source_local": {"trust": 0.95, "prefixes": ["local://"]},
}

def _load_db() -> Dict[str, Any]:
    if not os.path.exists(SRC_DB):
        return {"sources": dict(_DEFAULTS)}
    with open(SRC_DB, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_db(db: Dict[str, Any]) -> None:
    with open(SRC_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def register_source(source_id: str, *, trust: float, prefixes: Optional[List[str]] = None) -> None:
    db = _load_db()
    db.setdefault("sources", {})
    db["sources"][source_id] = {"trust": float(trust), "prefixes": list(prefixes or [])}
    _save_db(db)

def set_source_trust(source_id: str, trust: float) -> None:
    db = _load_db()
    if source_id not in db.get("sources", {}):
        db.setdefault("sources", {})[source_id] = {"trust": float(trust), "prefixes": []}
    else:
        db["sources"][source_id]["trust"] = float(trust)
    _save_db(db)

def _match_source(url: str) -> str:
    db = _load_db()
    for sid, rec in db.get("sources", {}).items():
        for p in rec.get("prefixes", []):
            if url.startswith(p):
                return sid
    # דיפולט: local אם לא פורמלי; אחרת נגזור domain־ish פשוט
    if url.startswith("local://"):
        return "source_local"
    # גזירת domain נאיבית (לוגיקה פשוטה כדי לא להכניס תלות)
    dom = url.split("://")[-1].split("/")[0].split("?")[0]
    return f"source_{dom or 'unknown'}"

def trust_for_url(url: str) -> float:
    db = _load_db()
    sid = _match_source(url)
    rec = db.get("sources", {}).get(sid)
    if rec is None:
        # מקורות לא ידועים – אמון בסיסי שמרני
        return 0.6
    return float(rec.get("trust", 0.6))

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sign_payload(payload: Any, *, secret: str) -> str:
    # חתימה דטרמיניסטית: sha256(secret || sha256(json))
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    inner = sha256_bytes(blob)
    return sha256_bytes((secret + inner).encode("utf-8"))

def normalize_and_sign(kind: str, info: Dict[str, Any], *, signing_secret: str) -> Dict[str, Any]:
    """
    מקבל אובייקט evidence כפי שמועבר ל-add_evidence ומחזיר אובייקט חתום ומנורמל.
    שדות חובה שנוסיף: ts, sha256, sig, source_id, source_trust.
    """
    now = float(time.time())
    src_url = str(info.get("source_url", "local://unknown"))
    src_trust = trust_for_url(src_url)
    ttl_s = float(info.get("ttl_s", 600.0))
    payload = info.get("payload", {})
    # sha256 של התוכן (payload בלבד – נתון לשינויי metadata)
    h = sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    sig = sign_payload(payload, secret=signing_secret)
    out = dict(info)
    out.update({
        "kind": kind,
        "ts": now,
        "ttl_s": ttl_s,
        "sha256": h,
        "sig": sig,
        "source_id": _match_source(src_url),
        "source_trust": float(src_trust),
    })
    return out

def is_fresh(ev: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    t = float(ev.get("ts", 0.0))
    ttl = float(ev.get("ttl_s", 0.0))
    return float(now or time.time()) <= t + ttl

def effective_session_trust(evidences: List[Dict[str, Any]], *, now: Optional[float] = None) -> float:
    """
    מחזיר אמון אפקטיבי לסשן: ממוצע משוקלל לפי:
      weight = source_trust * freshness_factor
      value  = min(source_trust, evidence_trust)
    """
    if not evidences:
        return 0.0
    now = float(now or time.time())
    num, den = 0.0, 0.0
    for ev in evidences:
        if not is_fresh(ev, now=now):
            continue
        # evidence_trust (שדה "trust") + source_trust
        e_tr = float(ev.get("trust", 0.5))
        s_tr = float(ev.get("source_trust", trust_for_url(str(ev.get("source_url","local://")))))
        # פקטור רעננות ליניארי פשוט (לא לפגוע בפשטות טסטים)
        age = max(0.0, now - float(ev.get("ts", now)))
        ttl = max(1.0, float(ev.get("ttl_s", 600.0)))
        fresh = max(0.0, 1.0 - (age/ttl))  # 1 כשהכי טרי, יורד עד 0
        weight = s_tr * fresh
        val = min(s_tr, e_tr)
        num += weight * val
        den += weight
    if den <= 1e-12:
        return 0.0
    return float(num/den)

