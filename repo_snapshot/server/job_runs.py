# server/job_runs.py
# רישום ריצות (job runs) וסטטיסטיקות: .imu/jobs/runs.jsonl (אירועים) + API סיכום.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json, time, statistics

J_DIR  = Path(".imu/jobs")
J_DIR.mkdir(parents=True, exist_ok=True)
RUNS_FILE = J_DIR / "runs.jsonl"

@dataclass
class JobRun:
    run_id: str
    kind: str
    ts_start: float
    ts_end: Optional[float] = None
    ok: Optional[bool] = None
    ms: Optional[int] = None
    meta: Dict[str, Any] = None

def _append(obj: Dict[str,Any]):
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def start_run(kind: str, meta: Dict[str,Any] | None = None) -> str:
    run_id = f"{kind}-{int(time.time()*1000)}"
    jr = JobRun(run_id=run_id, kind=kind, ts_start=time.time(), meta=meta or {})
    _append(asdict(jr))
    return run_id

def end_run(run_id: str, ok: bool, ms: int, extra: Dict[str,Any] | None = None):
    rec = {"run_id": run_id, "ts_end": time.time(), "ok": ok, "ms": ms}
    if extra: rec["extra"]=extra
    _append(rec)

def _iter_recent(hours: int) -> List[Dict[str,Any]]:
    horizon = time.time() - hours*3600
    out=[]
    if not RUNS_FILE.exists(): return out
    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj=json.loads(line)
                # כל run נכתב בשתי רשומות (start + end). נאחד לפי run_id
                out.append(obj)
            except Exception:
                continue
    # איחוד: build map run_id
    runs: Dict[str,Dict[str,Any]]={}
    for ev in out:
        rid = ev.get("run_id")
        if not rid: continue
        r = runs.setdefault(rid, {"run_id":rid, "kind":None, "ts_start":None, "ts_end":None, "ok":None, "ms":None})
        if "ts_start" in ev:
            r["ts_start"]=ev["ts_start"]; r["kind"]=ev.get("kind")
        if "ts_end" in ev:
            r["ts_end"]=ev["ts_end"]; r["ok"]=ev.get("ok"); r["ms"]=ev.get("ms")
    # סינון לפי זמן
    final=[]
    for r in runs.values():
        ts = r["ts_end"] or r["ts_start"] or 0
        if ts >= horizon:
            final.append(r)
    return final

def summary(hours: int = 24) -> Dict[str,Any]:
    recs=_iter_recent(hours)
    by_kind: Dict[str, List[Dict[str,Any]]] = {}
    for r in recs:
        by_kind.setdefault(r.get("kind") or "unknown", []).append(r)
    out={}
    for k, arr in by_kind.items():
        ms=[r["ms"] for r in arr if r.get("ms") is not None]
        ok=sum(1 for r in arr if r.get("ok") is True)
        fail=sum(1 for r in arr if r.get("ok") is False)
        total=len(arr)
        if ms:
            ms_sorted=sorted(ms)
            def q(p): 
                idx=int(p*(len(ms_sorted)-1)); 
                return ms_sorted[idx]
            metrics={"count": total, "ok": ok, "fail": fail,
                     "avg_ms": int(sum(ms)/len(ms)),
                     "p50_ms": int(q(0.50)), "p90_ms": int(q(0.90)), "p95_ms": int(q(0.95)), "p99_ms": int(q(0.99))}
        else:
            metrics={"count": total, "ok": ok, "fail": fail, "avg_ms": 0, "p50_ms":0, "p90_ms":0, "p95_ms":0, "p99_ms":0}
        out[k]=metrics
    return {"ok": True, "hours": hours, "kinds": out}