# server/canary_auto_api.py
# Auto-Canary: create → loop (probe+evaluate+step/promote/rollback) → stop/status
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time, threading, json, urllib.request, urllib.error, shutil

from server.canary_controller import deploy as canary_deploy, step as canary_step, promote as canary_promote, rollback as canary_rollback
from server.stream_wfq import BROKER
from policy.rbac import require_perm
from .canary_auto_policy import AutoCanaryPolicy

router = APIRouter(prefix="/auto_canary", tags=["auto-canary"])

def have(x:str)->bool: return shutil.which(x) is not None
def _emit(note:str, pct: Optional[float]=None, priority:int=4, topic:str="timeline"):
    BROKER.ensure_topic(topic, rate=100, burst=500, weight=2)
    ev={"type": "progress" if pct is not None else "event", "ts": time.time(), "note": note}
    if pct is not None: ev["pct"]=pct
    BROKER.submit(topic,"auto-canary",ev, priority=priority)

class StartReq(BaseModel):
    user_id: str = "demo-user"
    namespace: str = "default"
    app: str = "imu-app"
    image: str
    total_replicas: int = Field(10, ge=1)
    canary_percent: int = Field(10, ge=0, le=100)
    probe_url: str = Field(..., min_length=5)     # e.g. http://svc/health
    policy: AutoCanaryPolicy = AutoCanaryPolicy()
    dry: bool = False

class StopReq(BaseModel):
    run_id: str

class Status(BaseModel):
    run_id: str
    state: str
    ok_cycles: int
    steps: int
    last_error_rate: float
    last_p95_ms: float
    started_ts: float
    stopped_ts: Optional[float] = None

_RUNS: Dict[str, Status] = {}
_THREADS: Dict[str, threading.Thread] = {}
_STOP: Dict[str, bool] = {}

def _probe(url: str, sample: int = 20, timeout: float = 2.0) -> Dict[str,Any]:
    errs=0; times=[]
    for _ in range(sample):
        t0=time.time()
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"imu-canary"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status>=400: errs+=1
        except Exception:
            errs+=1
        dt=(time.time()-t0)*1000.0
        times.append(dt)
    times.sort()
    p95=times[int(0.95*(len(times)-1))] if times else 0.0
    return {"error_rate": errs/float(sample), "p95_ms": p95}

def _loop(run_id: str, req: StartReq):
    pol = req.policy
    st = Status(run_id=run_id, state="starting", ok_cycles=0, steps=0, last_error_rate=1.0, last_p95_ms=9999.0, started_ts=time.time())
    _RUNS[run_id]=st
    try:
        # 1) initial deploy
        _emit(f"auto_canary[{run_id}] deploy baseline/canary {req.total_replicas}/{req.canary_percent}%", pct=5)
        dres = canary_deploy(req) if not req.dry else {"ok": True}
        if not dres.get("ok"):
            st.state="failed"; _emit(f"auto_canary[{run_id}] deploy failed", priority=2); return
        st.state="running"; _RUNS[run_id]=st

        # 2) loop
        while not _STOP.get(run_id):
            # probe
            measures=_probe(req.probe_url)
            st.last_error_rate=measures["error_rate"]; st.last_p95_ms=measures["p95_ms"]; _RUNS[run_id]=st
            _emit(f"probe err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms")

            ok = (st.last_error_rate <= pol.max_error_rate) and (st.last_p95_ms <= pol.max_p95_ms)
            if ok:
                st.ok_cycles += 1
                _emit(f"OK cycle {st.ok_cycles}/{pol.consecutive_ok_for_promote}")
                # promote?
                if st.ok_cycles >= pol.consecutive_ok_for_promote:
                    p = canary_promote({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry}) if not req.dry else {"ok":True}
                    st.state="promoted"; _emit("promoted", pct=100, priority=3); break
                # step forward if steps cap not reached
                if st.steps < pol.max_steps:
                    st.steps += 1
                    canary_step({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"add_percent":pol.step_percent,"total_replicas":req.total_replicas,"dry":req.dry})
            else:
                # violation → rollback
                _emit(f"violation: err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms", priority=2)
                st.state="violated"
                if pol.rollback_on_first_violation:
                    canary_rollback({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                    st.state="rolled_back"; _emit("rolled_back", priority=2); break
                st.ok_cycles = 0
            # sleep and continue
            for _ in range(pol.hold_seconds):
                if _STOP.get(run_id): break
                time.sleep(1.0)
        if _STOP.get(run_id): st.state="stopped"
    except Exception as e:
        st.state=f"error:{e}"
    finally:
        st.stopped_ts=time.time(); _RUNS[run_id]=st; _STOP.pop(run_id, None)

@router.post("/start")
def start(req: StartReq):
    require_perm(req.user_id, "canary:auto:start")
    run_id=f"auto-{int(time.time())}"
    if _THREADS.get(run_id): raise HTTPException(409, "run_id exists")
    _STOP[run_id]=False
    t=threading.Thread(target=_loop, args=(run_id, req), daemon=True)
    _THREADS[run_id]=t; t.start()
    _emit(f"auto_canary[{run_id}] started", pct=1)
    return {"ok": True, "run_id": run_id}

@router.post("/stop")
def stop(req: StopReq):
    _STOP[req.run_id]=True
    return {"ok": True}

@router.get("/status")
def status(run_id: Optional[str] = None):
    if run_id: return {"ok": True, "status": _RUNS.get(run_id)}
    return {"ok": True, "runs": list(_RUNS.values())}