# imu_repo/audit/ledger_sync.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, hashlib, time, shutil
from audit.ledger import LEDGER_ROOT, LEDGER_FILE, verify_chain
from audit.signing import verify as verify_sig

SYNC_ROOT = "/mnt/data/imu_repo/audit_sync"  # מאגר ביניים לגרסאות מרוחקות

def _read_lines(path: str) -> List[Dict[str,Any]]:
    if not os.path.exists(path): return []
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            out.append(json.loads(ln))
    return out

def _write_lines(path: str, entries: List[Dict[str,Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False, separators=(",",":"))+"\n")

def export_snapshot(node_id: str) -> str:
    """
    מייצא צילום לדג’ר לקובץ snapshot עבור node_id (לשיתוף/העתקה ידנית/רשתית).
    """
    entries = _read_lines(LEDGER_FILE)
    snap = os.path.join(SYNC_ROOT, f"ledger_{node_id}.jsonl")
    _write_lines(snap, entries)
    return snap

def _hash_entry_core(e: Dict[str,Any]) -> str:
    core = {"_ts":e["_ts"], "_prev":e["_prev"], **{k:v for k,v in e.items() if not k.startswith("_")}}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode("utf-8")).hexdigest()

def _valid_link(prev_hash: str | None, e: Dict[str,Any]) -> bool:
    return (e.get("_prev")==prev_hash) and (e.get("_hash")==_hash_entry_core(e))

def import_and_merge(snapshot_path: str, *, require_hmac: bool=False, hmac_field: str="_sig") -> Dict[str,Any]:
    """
    ממזג צילום מרוחק לתוך הלדג’ר המקומי:
      - מוודא שלמות שרשרת (_prev/_hash)
      - אם require_hmac=True: בודק חתימה סימטרית על גוף האירוע (שדות ללא "_")
      - אסטרטגיית fork: "Longest valid chain wins" עם חותמת זמן אחרונה גבוהה יותר כשווה אורך.
    מחזיר {"merged": n, "fork_resolved": bool}
    """
    remote = _read_lines(snapshot_path)
    local  = _read_lines(LEDGER_FILE)
    if not remote: return {"merged":0,"fork_resolved":False}

    # אימות לכל הרשומות המרוחקות כשרשרת
    prev = None
    for e in remote:
        if require_hmac:
            core = {k:v for k,v in e.items() if not k.startswith("_") and k!=hmac_field}
            if not verify_sig(core, e.get(hmac_field,"")):
                raise RuntimeError("hmac_verify_failed")
        if not _valid_link(prev, e):
            raise RuntimeError("remote_chain_invalid")
        prev = e["_hash"]

    # אם המקומי ריק — נכתוב את המרוחק
    if not local:
        _write_lines(LEDGER_FILE, remote)
        return {"merged": len(remote), "fork_resolved": False}

    # בדיקת fork: נמצא נקודת מפגש earliest
    i=j=0
    # מצא prefix מקומי שמוכל במרוחק
    local_hashes = [e["_hash"] for e in local]
    remote_hashes= [e["_hash"] for e in remote]
    # אם המקומי סיומת של המרוחק — החלפה
    if len(remote) >= len(local) and local_hashes == remote_hashes[:len(local)]:
        _write_lines(LEDGER_FILE, remote)
        return {"merged": len(remote)-len(local), "fork_resolved": False}
    # אם המרוחק סיומת של המקומי — אין מה לעשות
    if len(local) >= len(remote) and remote_hashes == local_hashes[:len(remote)]:
        return {"merged":0,"fork_resolved": False}

    # אחרת: יש פיצול. נחתוך לנקודת LCA (ה-prefix המשותף הארוך ביותר)
    k=0
    L=min(len(local), len(remote))
    while k<L and local[k]["_hash"]==remote[k]["_hash"]:
        k+=1
    # prefix משותף = k רשומות ראשונות
    common = local[:k]
    # בוחרים שרשרת "טובה יותר": ארוכה יותר; ואם אורך שווה — לפי _ts אחרון גבוה יותר
    cand_local  = local
    cand_remote = remote
    def score(chain): 
        return (len(chain), chain[-1]["_ts"] if chain else 0.0)
    best = max([cand_local, cand_remote], key=score)
    _write_lines(LEDGER_FILE, best)
    ok = verify_chain()
    if not ok: 
        raise RuntimeError("merged_chain_invalid")
    return {"merged": abs(len(cand_remote)-len(cand_local)), "fork_resolved": True}