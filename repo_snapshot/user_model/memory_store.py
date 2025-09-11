# imu_repo/user_model/memory_store.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, time, math, json, hashlib
from user_model.identity import user_dir, load_key
from user_model.crypto_store import save_encrypted_json, load_encrypted_json

MEM_FILE = "mem.json.enc"
NONCE = b"IMU_MEM_V1__NONCE"  # nonce קבוע לקובץ (הצפנה במנוחה; לא תקשורת)

def _path(uid: str) -> str:
    return os.path.join(user_dir(uid), MEM_FILE)

def _now() -> float: return time.time()

def _new_doc() -> Dict[str,Any]:
    return {"T0": [], "T1": {}, "T2": {}, "log": []}
    # T0: אירועים אפיזודיים
    # T1: תכונות/העדפות קצרות טווח (מיפוי -> {"mu":p, "n":count, "last_ts":...})
    # T2: אמונות/מטרות/תרבות ארוכות טווח (כנ"ל)

def load(uid: str) -> Dict[str,Any]:
    p = _path(uid)
    if not os.path.exists(p):
        return _new_doc()
    key = load_key(uid)
    try:
        return load_encrypted_json(p, key, nonce=NONCE)
    except Exception:
        # שחזור סובלני
        return _new_doc()

def save(uid: str, doc: Dict[str,Any]) -> None:
    key = load_key(uid)
    os.makedirs(user_dir(uid), exist_ok=True)
    save_encrypted_json(_path(uid), key, doc, nonce=NONCE)

def put_event(uid: str, kind: str, key: str, value: Any, *, confidence: float=0.7, ttl_s: float=90*24*3600, source: str="user", evidence_id: str | None=None) -> None:
    """
    מוסיף אירוע ל-T0 ומעדכן T1/T2 בהטיה לפי recency*confidence.
    kind ∈ {"pref","belief","affect","goal"} → pref→T1, אחרים→T2
    """
    doc = load(uid)
    ev = {"ts": _now(), "kind": kind, "key": key, "value": value, "conf": float(confidence), "ttl_s": float(ttl_s), "source": source, "evidence_id": evidence_id}
    doc["T0"].append(ev)

    target = "T1" if kind=="pref" else "T2"
    slot = doc[target].get(key) or {"mu": None, "n": 0, "last_ts": 0.0, "sources": []}

    # המרה לערך מספרי הסתברותי בסיסי: בוליאני → {True:1, False:0}, מספר: זהה; טקסט: hash→[0..1]
    def to_num(v: Any) -> float:
        if isinstance(v, bool): return 1.0 if v else 0.0
        if isinstance(v, (int, float)): return float(v)
        h = int(hashlib.sha256(str(v).encode("utf-8")).hexdigest(), 16)
        return (h % 1000) / 1000.0

    x = to_num(value)
    age_s = max(1.0, _now() - ev["ts"])
    rec_weight = 1.0 / math.log(10.0 + age_s)   # דעיכה איטית בזמן
    w = max(0.01, float(confidence)) * rec_weight

    # עדכון אומדן "ממוצע משוקלל"
    if slot["mu"] is None:
        slot["mu"] = x
        slot["n"] = 1
    else:
        slot["mu"] = (slot["mu"]*slot["n"] + x*w) / (slot["n"] + w)
        slot["n"]  = slot["n"] + w
    slot["last_ts"] = ev["ts"]
    slot["sources"].append({"source": source, "evidence_id": evidence_id, "ts": ev["ts"], "conf": confidence})
    doc[target][key] = slot

    save(uid, doc)

def garbage_collect(uid: str) -> None:
    """TTL ל-T0 + איחוד T1/T2 (דחיסה קלה)."""
    doc = load(uid)
    now = _now()
    T0 = []
    for ev in doc["T0"]:
        if now - ev["ts"] <= ev.get("ttl_s", 0):
            T0.append(ev)
    doc["T0"] = T0
    # ניתן להוסיף כאן דחיסה/סף n מינימלי
    save(uid, doc)

def get_profile(uid: str) -> Dict[str,Any]:
    """פרופיל מרוכז כתמונת 'תודעה' מעשית (סטייט החלטות)."""
    doc = load(uid)
    out = {
        "pref": {k: v["mu"] for k,v in doc["T1"].items()},
        "beliefs": {k: v["mu"] for k,v in doc["T2"].items()},
        "strength": {k: v["n"] for k,v in {**doc["T1"], **doc["T2"]}.items()}
    }
    return out

def forget(uid: str) -> None:
    """מחיקה לוגית: ריקון הזיכרון (בנוסף למחיקה קשיחה ב-identity.forget_user אם יידרש)."""
    save(uid, _new_doc())


def list_events(uid: str) -> List[Dict[str,Any]]:
    """מחזיר העתק של T0 (כולל שדה id מחושב אם חסר)."""
    from user_model.event_crdt import event_id
    doc = load(uid)
    out=[]
    for ev in doc["T0"]:
        ev2 = dict(ev)
        ev2["id"] = ev.get("id") or event_id(ev2)
        out.append(ev2)
    return out


def rebuild_from_events(uid: str, events: List[Dict[str,Any]]) -> None:
    """
    בונה מחדש T1/T2 מאירועי T0 (איחוד מלא; GC לא מתבצע כאן).
    """
    # אפס את המסמך ושחזר
    doc = {"T0": [], "T1": {}, "T2": {}, "log": []}
    save(uid, doc)
    for ev in sorted(events, key=lambda e: float(e.get("ts",0.0))):
        put_event(uid, ev.get("kind","pref"), ev.get("key",""), ev.get("value"),
                  confidence=float(ev.get("conf",0.7)),
                  ttl_s=float(ev.get("ttl_s", 90*24*3600)),
                  source=ev.get("source","import"),
                  evidence_id=ev.get("evidence_id"))