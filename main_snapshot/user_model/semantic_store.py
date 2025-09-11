# imu_repo/user_model/semantic_store.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, json, math, re, time, hashlib
from user_model.identity import user_dir, ensure_master_key
from user_model.crypto_utils import seal, open_sealed
from user_model.consent import check as check_consent

MEM_ROOT = "mem"          # תחת user_dir
INDEX   = "index.json"    # מטה-דאטה (ללא טקסט מקור)
BLOBS   = "blobs"         # מטען מוצפן (טקסט/JSON)

TOKEN = re.compile(r"[A-Za-zא-ת0-9]+")

def _vec(text: str) -> Dict[str, float]:
    # bag-of-words מנורמל
    toks = [t.lower() for t in TOKEN.findall(text)]
    if not toks: return {}
    freq: Dict[str,int] = {}
    for t in toks: freq[t] = freq.get(t,0)+1
    n = float(sum(freq.values()))
    return {k:(v/n) for k,v in freq.items()}

def _cos(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k]*b[k] for k in keys)
    na = math.sqrt(sum(x*x for x in a.values()))
    nb = math.sqrt(sum(x*x for x in b.values()))
    if na==0.0 or nb==0.0: return 0.0
    return dot/(na*nb)

def _paths(user_key: str) -> Dict[str,str]:
    up = user_dir(user_key)
    root = os.path.join(up, MEM_ROOT)
    os.makedirs(os.path.join(root,BLOBS), exist_ok=True)
    return {"root": root, "index": os.path.join(root, INDEX), "blobs": os.path.join(root,BLOBS)}

def _load_index(p: str) -> List[Dict[str,Any]]:
    try:
        return json.load(open(p,"r",encoding="utf-8"))
    except Exception:
        return []

def _save_index(p: str, arr: List[Dict[str,Any]]) -> None:
    json.dump(arr, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def _seal_text(master: bytes, text: str) -> str:
    return seal(text.encode("utf-8"), master)

def _open_text(master: bytes, payload: str) -> str:
    return open_sealed(payload, master).decode("utf-8")

def add_memory(user_key: str, *, text: str, kind: str="note", purpose: str="preferences",
               tier: str="T1", confidence: float=0.6, ttl_s: int=365*24*3600) -> str:
    """
    כותב פריט זיכרון מוצפן ל־blobs ושומר מטריצה דלה ב-index.
    דורש consent לפורפוס.
    """
    if not check_consent(user_key, purpose).get("ok", False):
        raise PermissionError("consent_required")

    paths = _paths(user_key)
    idx = _load_index(paths["index"])
    master = ensure_master_key(user_key)

    payload = _seal_text(master, text)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    blob_p = os.path.join(paths["blobs"], sha+".json")
    with open(blob_p,"w",encoding="utf-8") as f: f.write(payload)

    meta = {
        "sha": sha, "kind": kind, "purpose": purpose, "tier": tier,
        "added_at": time.time(), "ttl_s": int(ttl_s), "confidence": float(confidence),
        "vec": _vec(text)  # דל — כדי שלא לחשוף תוכן מלא ב-index
    }
    idx.append(meta); _save_index(paths["index"], idx)
    return sha

def get_memory(user_key: str, sha: str) -> Dict[str,Any]:
    paths = _paths(user_key)
    idx = _load_index(paths["index"])
    rec = next((r for r in idx if r["sha"]==sha), None)
    if not rec: raise KeyError("not_found")
    master = ensure_master_key(user_key)
    blob_p = os.path.join(paths["blobs"], sha+".json")
    text = _open_text(master, open(blob_p,"r",encoding="utf-8").read())
    return {"meta": rec, "text": text}

def search(user_key: str, query: str, *, topk: int=5, purpose: str | None=None) -> List[Tuple[float, Dict[str,Any]]]:
    paths = _paths(user_key)
    idx = _load_index(paths["index"])
    qv = _vec(query)
    scored=[]
    now = time.time()
    for rec in idx:
        if purpose and rec["purpose"] != purpose: continue
        if now > rec["added_at"] + rec["ttl_s"]: 
            continue  # פג
        s = _cos(rec.get("vec",{}), qv)
        # היסט העדפה לפי confidence (משקף ToM לייט)
        s = s * (0.5 + 0.5*float(rec.get("confidence",0.5)))
        scored.append((s, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]

def consolidate(user_key: str, *, min_hits:int=2, promote_confidence: float=0.2) -> Dict[str,int]:
    """
    מעלה T1 ל־T2 כאשר יש חזרות/חיזוקים.
    """
    paths = _paths(user_key)
    idx = _load_index(paths["index"])
    counts: Dict[str,int] = {}
    for rec in idx:
        if rec["tier"]=="T1":
            key = (rec["kind"], rec["purpose"])
            k = f"{key}"
            counts[k] = counts.get(k,0)+1
    promoted=0
    for rec in idx:
        if rec["tier"]=="T1":
            k = f"{(rec['kind'],rec['purpose'])}"
            if counts.get(k,0) >= min_hits:
                rec["tier"]="T2"
                rec["confidence"]=min(1.0, float(rec.get("confidence",0.5))+promote_confidence)
                promoted += 1
    _save_index(paths["index"], idx)
    return {"promoted": promoted}