# imu_repo/dist/job_queue.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Callable
import os, json, time, uuid, hashlib

ROOT = "/mnt/data/imu_repo/queue"
DIRS = ["queued","reserved","done","failed","dedupe"]
WAL  = os.path.join(ROOT, "wal.jsonl")
LEASE_S = 10.0

def _ensure():
    os.makedirs(ROOT, exist_ok=True)
    for d in DIRS:
        os.makedirs(os.path.join(ROOT,d), exist_ok=True)

def _now() -> float: return time.time()

def _write(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)
    os.replace(tmp, path)

def _read(path: str) -> Any:
    if not os.path.exists(path): return None
    return json.load(open(path,"r",encoding="utf-8"))

def _wal_write(ev: Dict[str,Any]) -> None:
    ev = {"ts": _now(), **ev}
    with open(WAL,"a",encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def _job_path(state: str, job_id: str) -> str:
    return os.path.join(ROOT, state, f"{job_id}.json")

def _dedupe_key(payload: Dict[str,Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

def enqueue(payload: Dict[str,Any], *, idempotency_key: Optional[str]=None) -> Dict[str,Any]:
    """
    מוסיף עבודה חדשה עם Idempotency:
      - אם יש idempotency_key שכבר בוצע/בתור — תחזיר מצבו וקישור ל-job_id הקיים.
      - אחרת תיצור רשומה חדשה ב-queued/ ותעדכן WAL.
    """
    _ensure()
    ik = idempotency_key or _dedupe_key(payload)
    dk = os.path.join(ROOT, "dedupe", ik + ".json")
    if os.path.exists(dk):
        info = _read(dk)
        return {"ok": True, "job_id": info["job_id"], "state": info["state"], "idempotent": True}
    job_id = uuid.uuid4().hex[:12]
    item = {"job_id": job_id, "payload": payload, "state": "queued", "created_at": _now()}
    _write(_job_path("queued", job_id), item)
    _write(dk, {"job_id": job_id, "state": "queued"})
    _wal_write({"op":"enqueue","job_id":job_id,"payload":payload,"ik":ik})
    return {"ok": True, "job_id": job_id, "state": "queued", "idempotent": False}

def reserve(lease_s: float=LEASE_S) -> Optional[Dict[str,Any]]:
    """
    מקצה עבודה (Lease): מעביר queued->reserved עם expires_at; אם אין — מחזיר None.
    """
    _ensure()
    for fn in sorted(os.listdir(os.path.join(ROOT,"queued"))):
        if not fn.endswith(".json"): continue
        job_id = fn[:-5]
        qpath = _job_path("queued", job_id)
        job = _read(qpath)
        if not job: continue
        job["state"]="reserved"; job["reserved_at"]=_now(); job["lease_until"]=_now()+lease_s
        _write(_job_path("reserved", job_id), job)
        os.remove(qpath)
        _wal_write({"op":"reserve","job_id":job_id,"lease_until":job["lease_until"]})
        return job
    # חידוש עבודות שפג להן lease (requeue)
    for fn in sorted(os.listdir(os.path.join(ROOT,"reserved"))):
        if not fn.endswith(".json"): continue
        job_id = fn[:-5]
        rpath = _job_path("reserved", job_id)
        job = _read(rpath)
        if not job: continue
        if _now() > job.get("lease_until",0):
            job["state"]="queued"; job.pop("reserved_at",None); job.pop("lease_until",None)
            _write(_job_path("queued", job_id), job)
            os.remove(rpath)
            _wal_write({"op":"requeue_expired","job_id":job_id})
            return job
    return None

def ack(job_id: str, result: Dict[str,Any] | None=None) -> None:
    _ensure()
    rpath = _job_path("reserved", job_id)
    job = _read(rpath)
    if not job: raise FileNotFoundError("job_not_reserved")
    job["state"]="done"; job["result"]=result or {"ok":True}
    _write(_job_path("done", job_id), job)
    os.remove(rpath)
    _wal_write({"op":"ack","job_id":job_id,"result":job["result"]})
    # עדכן dedupe
    # מצא מפתח דה-דופ (אם יש): חפש בפיילי WAL האחרון של enqueue
    # לשמירה על פשטות – לא נחלץ כאן; הדה-דופ נשאר מצבני לפי ההכנסה

def nack(job_id: str, reason: str, *, compensate: Dict[str,Any] | None=None) -> None:
    """
    מסמן כ-failed ומבצע פיצוי (Rollback) דטרמיניסטי (אם הוגדר).
      compensate:
        {"type":"delete_file","path": "..."}   # מוחק קובץ אם קיים
        {"type":"noop"}
    """
    _ensure()
    rpath = _job_path("reserved", job_id)
    job = _read(rpath)
    if not job: raise FileNotFoundError("job_not_reserved")
    # פיצוי
    if compensate:
        if compensate.get("type")=="delete_file":
            p = compensate.get("path")
            try:
                if p and os.path.exists(p): os.remove(p)
            except Exception: pass
        # noop: לא צריך לעשות דבר
    job["state"]="failed"; job["error"]=reason; job["compensate"]=compensate
    _write(_job_path("failed", job_id), job)
    os.remove(rpath)
    _wal_write({"op":"nack","job_id":job_id,"reason":reason,"compensate":compensate})

def replay_from_wal(clear_first: bool=False) -> Dict[str,int]:
    """
    בונה מחדש את מצב התור מ-WAL (במקרה של קריסה).
    """
    _ensure()
    if clear_first:
        for d in DIRS:
            for fn in os.listdir(os.path.join(ROOT,d)):
                os.remove(os.path.join(ROOT,d,fn))
    stats = {"enqueue":0,"reserve":0,"requeue_expired":0,"ack":0,"nack":0}
    if not os.path.exists(WAL): return stats
    with open(WAL,"r",encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            op = ev.get("op")
            if op=="enqueue":
                stats["enqueue"]+=1
                job_id = ev["job_id"]
                payload = ev["payload"]
                item = {"job_id": job_id, "payload": payload, "state": "queued", "created_at": ev.get("ts", _now())}
                _write(_job_path("queued", job_id), item)
            elif op=="reserve":
                stats["reserve"]+=1
                job_id = ev["job_id"]
                qpath = _job_path("queued", job_id)
                job = _read(qpath)
                if job:
                    job["state"]="reserved"; job["reserved_at"]=ev.get("ts",_now()); job["lease_until"]=ev.get("lease_until", _now()+LEASE_S)
                    _write(_job_path("reserved", job_id), job)
                    os.remove(qpath)
            elif op=="requeue_expired":
                stats["requeue_expired"]+=1
                job_id = ev["job_id"]
                rpath = _job_path("reserved", job_id)
                job = _read(rpath)
                if job:
                    job["state"]="queued"; job.pop("reserved_at",None); job.pop("lease_until",None)
                    _write(_job_path("queued", job_id), job)
                    os.remove(rpath)
            elif op=="ack":
                stats["ack"]+=1
                job_id = ev["job_id"]
                rpath = _job_path("reserved", job_id)
                job = _read(rpath)
                if job:
                    job["state"]="done"; job["result"]=ev.get("result",{"ok":True})
                    _write(_job_path("done", job_id), job)
                    os.remove(rpath)
            elif op=="nack":
                stats["nack"]+=1
                job_id = ev["job_id"]
                rpath = _job_path("reserved", job_id)
                job = _read(rpath)
                if job:
                    job["state"]="failed"; job["error"]=ev.get("reason",""); job["compensate"]=ev.get("compensate")
                    _write(_job_path("failed", job_id), job)
                    os.remove(rpath)
    return stats