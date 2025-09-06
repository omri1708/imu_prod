# server/scheduler_api.py
# Built-in scheduler (no external deps): at/interval schedules; tasks: runbook.unity_k8s, auto_canary.start,
# gatekeeper.evaluate_and_set_status, emit_event (for test). Persists to .imu/schedules/schedules.json
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time, threading, uuid, urllib.request

from policy.rbac import require_perm
from server.stream_wfq import BROKER

router = APIRouter(prefix="/scheduler", tags=["scheduler"])

S_DIR  = Path(".imu/schedules"); S_DIR.mkdir(parents=True, exist_ok=True)
S_FILE = S_DIR / "schedules.json"

_LOCK = threading.Lock()
_SCHED: Dict[str,dict] = {}
_THREAD: Optional[threading.Thread] = None
_STOP = False

def _save():
    with _LOCK:
        S_FILE.write_text(json.dumps(_SCHED, ensure_ascii=False, indent=2), encoding="utf-8")

def _load():
    global _SCHED
    if S_FILE.exists():
        try:
            _SCHED = json.loads(S_FILE.read_text(encoding="utf-8"))
        except Exception:
            _SCHED = {}
    else:
        _SCHED = {}

def _emit(note:str, topic:str="timeline", pct:float|None=None, priority:int=4):
    BROKER.ensure_topic(topic, rate=100, burst=500, weight=2)
    ev={"type":"progress" if pct is not None else "event", "ts": time.time(), "note":note}
    if pct is not None: ev["pct"]=pct
    BROKER.submit(topic,"scheduler",ev, priority=priority)

def _call(path:str, body:dict)->dict:
    req=urllib.request.Request("http://127.0.0.1:8000"+path, method="POST",
                               data=json.dumps(body).encode("utf-8"),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _run_task(kind:str, args:dict):
    """Execute single scheduled task kind."""
    if kind=="emit_event":
        _emit(args.get("note","scheduled event"), topic=args.get("topic","timeline"), pct=args.get("pct"))
        return
    if kind=="runbook.unity_k8s":
        _call("/runbook/unity_k8s", args)
        return
    if kind=="auto_canary.start":
        _call("/auto_canary/start", args)
        return
    if kind=="gatekeeper.evaluate_and_set_status":
        _call("/gatekeeper/evaluate_and_set_status", args)
        return
    # extend here as needed
    raise RuntimeError(f"unknown task kind {kind}")

def _worker():
    global _STOP
    while not _STOP:
        now=time.time()
        due=[]
        with _LOCK:
            for sid, it in list(_SCHED.items()):
                if it.get("disabled"): continue
                nxt = float(it.get("next_run_ts", 0.0))
                if now >= nxt:
                    due.append((sid, it))
        for sid, it in due:
            try:
                _emit(f"schedule[{sid}] start: {it['kind']}", priority=5)
                _run_task(it["kind"], it.get("args",{}))
                _emit(f"schedule[{sid}] done", priority=5)
            except Exception as e:
                _emit(f"schedule[{sid}] error: {e}", priority=2)
            # reschedule
            with _LOCK:
                if sid not in _SCHED: continue
                it2=_SCHED[sid]
                if it2.get("mode")=="interval":
                    it2["next_run_ts"] = now + float(it2.get("interval_s", 60.0))
                else:
                    it2["disabled"] = True
                _save()
        time.sleep(0.5)

def scheduler_boot():
    global _THREAD, _STOP
    _load()
    _STOP=False
    if _THREAD and _THREAD.is_alive(): return
    _THREAD=threading.Thread(target=_worker, name="imu-scheduler", daemon=True)
    _THREAD.start()

def scheduler_stop():
    global _STOP
    _STOP=True

class CreateReq(BaseModel):
    user_id: str = "demo-user"
    kind: str = Field(..., description="emit_event | runbook.unity_k8s | auto_canary.start | gatekeeper.evaluate_and_set_status")
    mode: str = Field("at", regex="^(at|interval)$")
    at_ts: Optional[float] = None
    interval_s: Optional[float] = None
    args: Dict[str,Any] = Field(default_factory=dict)
    disabled: bool = False

@router.post("/create")
def create(req: CreateReq):
    require_perm(req.user_id, "scheduler:create")
    sid = uuid.uuid4().hex[:12]
    if req.mode=="at":
        next_ts = float(req.at_ts or (time.time()+5.0))
    else:
        if not req.interval_s or req.interval_s<=0: raise HTTPException(400,"interval_s required >0")
        next_ts = time.time() + float(req.interval_s)
    with _LOCK:
        _SCHED[sid] = {"id":sid,"kind":req.kind,"mode":req.mode,"interval_s":req.interval_s,
                       "args":req.args,"next_run_ts":next_ts,"disabled":bool(req.disabled)}
        _save()
    return {"ok": True, "id": sid}

@router.get("/list")
def list_sched():
    with _LOCK: return {"ok": True, "items": list(_SCHED.values())}

class DeleteReq(BaseModel):
    user_id: str = "demo-user"
    id: str

@router.post("/delete")
def delete(req: DeleteReq):
    require_perm(req.user_id, "scheduler:delete")
    with _LOCK:
        if req.id in _SCHED: _SCHED.pop(req.id); _save()
    return {"ok": True}

@router.post("/start")
def start_api():
    scheduler_boot(); return {"ok": True}

@router.post("/stop")
def stop_api():
    scheduler_stop(); return {"ok": True}

@router.get("/state")
def state():
    with _LOCK:
        return {"ok": True, "count": len(_SCHED)}