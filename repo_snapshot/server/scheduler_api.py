# server/scheduler_api.py
# Built-in scheduler (no external deps): at/interval schedules; tasks:
#  - emit_event
#  - runbook.unity_k8s
#  - auto_canary.start
#  - gatekeeper.evaluate_and_set_status
#  - supplychain.attest                  (NEW)
#  - unified.export_signed               (NEW)
# Data persisted to .imu/schedules/schedules.json

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time, threading, uuid, urllib.request, urllib.parse

from policy.rbac import require_perm
from server.stream_wfq import BROKER
from server.job_runs import start_run, end_run

router = APIRouter(prefix="/scheduler", tags=["scheduler"])

S_DIR  = Path(".imu/schedules"); S_DIR.mkdir(parents=True, exist_ok=True)
S_FILE = S_DIR / "schedules.json"

_LOCK = threading.Lock()
_SCHED: Dict[str,dict] = {}
_THREAD: Optional[threading.Thread] = None
_STOP = False

SUPPORTED_TASKS = {
    "emit_event": {
        "desc": "Publish an event into WFQ timeline",
        "args": {"topic":"timeline", "note":"text", "pct":"optional number"}
    },
    "runbook.unity_k8s": {
        "desc": "Runbook: Unity→K8s (see /runbook/unity_k8s for args)",
        "args": {"user_id":"demo-user","project_dir":"/path","target":"Android","namespace":"default","name":"unity-app"}
    },
    "auto_canary.start": {
        "desc": "Start Auto-Canary (see /auto_canary/start)",
        "args": {"user_id":"demo-user","namespace":"default","app":"imu-app","image":"nginx:alpine","total_replicas":10,"canary_percent":10,"probe_url":"http://..."}
    },
    "gatekeeper.evaluate_and_set_status": {
        "desc": "Run Gatekeeper + set GitHub status",
        "args": {"evidences":[],"checks":None,"p95":None,"owner":"org","repo":"repo","sha":"<commit>","context":"IMU/Gatekeeper"}
    },
    "supplychain.attest": {
        "desc": "cosign attest SBOM for docker image",
        "args": {"image":"nginx:alpine", "predicate_path":"sbom/cyclonedx_demo.json"}
    },
    "unified.export_signed": {
        "desc": "Export unified archive (ZIP + DSSE), name=<timestamped>",
        "args": {"name":"nightly"}
    }
}

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

def _http_call(method: str, path: str, body: dict | None = None) -> dict:
    url = "http://127.0.0.1:8000" + path
    if method == "GET" and body:
        # append query
        q = urllib.parse.urlencode(body)
        url = url + ("&" if "?" in url else "?") + q
        req = urllib.request.Request(url, method="GET", headers={"User-Agent":"imu-scheduler"})
    else:
        req = urllib.request.Request(url, method=method,
                                     data=(json.dumps(body).encode("utf-8") if body is not None else None),
                                     headers={"Content-Type":"application/json","User-Agent":"imu-scheduler"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.headers.get_content_type() == "application/json":
            return json.loads(r.read().decode())
        return {"ok": True, "bytes": len(r.read())}

def _run_task(kind:str, args:dict):
    run_id = start_run(kind, meta=args or {})
    t0 = time.time()
    try:
        if kind == "emit_event":
            _emit(args.get("note","scheduled event"), topic=args.get("topic","timeline"), pct=args.get("pct")); ok=True
        elif kind == "runbook.unity_k8s":
            _http_call("POST","/runbook/unity_k8s", args); ok=True
        elif kind == "auto_canary.start":
            _http_call("POST","/auto_canary/start", args); ok=True
        elif kind == "gatekeeper.evaluate_and_set_status":
            _http_call("POST","/gatekeeper/evaluate_and_set_status", args); ok=True
        elif kind == "supplychain.attest":
            _http_call("POST","/supplychain/index/attest", args); ok=True
        elif kind == "unified.export_signed":
            name = args.get("name") or f"nightly_{int(time.time())}"
            _http_call("GET", "/unified/export_signed", {"name": name}); ok=True
        else:
            raise RuntimeError(f"unknown task kind {kind}")
    except Exception:
        ok=False
        raise
    finally:
        ms=int((time.time()-t0)*1000)
        end_run(run_id, ok=ok, ms=ms, extra={"args": args})

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

# boot/stop
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

# --------- API ---------
@router.get("/tasks")
def tasks():
    return {"ok": True, "tasks": SUPPORTED_TASKS}

class CreateReq(BaseModel):
    user_id: str = "demo-user"
    kind: str = Field(..., description="see /scheduler/tasks")
    mode: str = Field("at", pattern="^(at|interval)$")
    at_ts: Optional[float] = None
    interval_s: Optional[float] = None
    args: Dict[str,Any] = Field(default_factory=dict)
    disabled: bool = False

@router.post("/create")
def create(req: CreateReq):
    require_perm(req.user_id, "scheduler:create")
    sid = uuid.uuid4().hex[:12]
    if req.kind not in SUPPORTED_TASKS:
        raise HTTPException(400, f"unsupported task kind {req.kind}")
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

class UpdateReq(BaseModel):
    user_id: str = "demo-user"
    id: str
    disabled: Optional[bool] = None
    next_run_ts: Optional[float] = None
    interval_s: Optional[float] = None
    args: Optional[Dict[str,Any]] = None

@router.post("/update")
def update(req: UpdateReq):
    require_perm(req.user_id, "scheduler:update")
    with _LOCK:
        it=_SCHED.get(req.id)
        if not it: raise HTTPException(404,"not found")
        if req.disabled is not None: it["disabled"]=bool(req.disabled)
        if req.next_run_ts is not None: it["next_run_ts"]=float(req.next_run_ts)
        if req.interval_s is not None: it["interval_s"]=float(req.interval_s)
        if req.args is not None: it["args"]=req.args
        _save()
    return {"ok": True}

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