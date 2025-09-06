# server/canary_auto_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time, threading, json, urllib.request, shutil

from server.canary_controller import deploy as canary_deploy, step as canary_step, promote as canary_promote, rollback as canary_rollback
from server.stream_wfq import BROKER
from policy.rbac import require_perm
from .canary_auto_policy import AutoCanaryPolicy
from .k8s_ready import readiness_ratio, have_kubectl
from .prometheus_client import query_instant, query_range, extract_last_vector, quantile_from_range
from .prom_anomaly import detect_spike, detect_lag_spike
from .gatekeeper_client import evaluate as gate_evaluate
from .decision_log import record_gate_decision

router = APIRouter(prefix="/auto_canary", tags=["auto-canary"])

def have(x:str)->bool: return shutil.which(x) is not None
def _emit(note:str, pct: Optional[float]=None, priority:int=4, topic:str="timeline"):
    BROKER.ensure_topic(topic, rate=100, burst=500, weight=2)
    ev={"type": "progress" if pct is not None else "event", "ts": time.time(), "note": note}
    if pct is not None: ev["pct"]=pct
    BROKER.submit(topic,"auto-canary",ev, priority=priority)

class GateChecks(BaseModel):
    owner: str
    repo: str
    ref: Optional[str] = None
    pr_number: Optional[int] = None
    required: list[str] = Field(default_factory=list)
    mode: str = Field("all", regex="^(all|any)$")
    token_env: str = "GITHUB_TOKEN"

class GateP95(BaseModel):
    keys: list[str]
    ceiling_ms: int = 5000

class GateInput(BaseModel):
    evidences: list[dict] = Field(default_factory=list)
    checks: Optional[GateChecks] = None
    p95: Optional[GateP95] = None

class PRStatus(BaseModel):
    owner: str
    repo: str
    sha: str
    context: str = "IMU/Auto-Canary"

class StartReq(BaseModel):
    user_id: str = "demo-user"
    namespace: str = "default"
    app: str = "imu-app"
    image: str
    total_replicas: int = Field(10, ge=1)
    canary_percent: int = Field(10, ge=0, le=100)
    probe_url: Optional[str] = None
    prom_url: Optional[str] = None
    q_error_rate: Optional[str] = None
    q_latency_ms: Optional[str] = None
    prom_window_s: int = 180
    policy: AutoCanaryPolicy = AutoCanaryPolicy()
    # Gatekeeper before promote:
    gatekeeper_required: bool = False
    gate: Optional[GateInput] = None
    gate_fail_rollback: bool = True
    # PR Status context (GitHub) לאורך הקאנרי:
    pr_status: Optional[PRStatus] = None
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
    last_ready_ratio: float
    started_ts: float
    stopped_ts: Optional[float] = None

_RUNS: Dict[str, Status] = {}
_THREADS: Dict[str, threading.Thread] = {}
_STOP: Dict[str, bool] = {}

def _http_probe(url: str, sample: int = 20, timeout: float = 2.0) -> Dict[str,Any]:
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

def _prom_series(prom_url: str, promql: str, window_s: int) -> list[float]:
    # מחזיר values מהטווח האחרון (samples כ-ms או rate)
    now=time.time()
    step=max(5, window_s//30)
    resp=query_range(prom_url, promql, start=now-window_s, end=now, step=step)
    series=resp.get("data",{}).get("result",[])
    if not series: return []
    vals=[float(v[1]) for v in series[0].get("values",[]) if v and v[1] is not None]
    return vals

def _status_set(prs: Optional[PRStatus], state: str, desc: str):
    if not prs: return
    try:
        url="http://127.0.0.1:8000/status/github/set"
        body={"user_id":"demo-user","owner":prs.owner,"repo":prs.repo,"sha":prs.sha,"state":state,"context":prs.context,"description":desc}
        req=urllib.request.Request(url, method="POST", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass

def _loop(run_id: str, req: StartReq):
    pol = req.policy
    st = Status(run_id=run_id, state="starting", ok_cycles=0, steps=0,
                last_error_rate=1.0, last_p95_ms=9999.0, last_ready_ratio=0.0, started_ts=time.time())
    _RUNS[run_id]=st
    try:
        _status_set(req.pr_status, "pending", "canary starting")
        _emit(f"auto_canary[{run_id}] deploy baseline/canary {req.total_replicas}/{req.canary_percent}%", pct=5)
        dres = canary_deploy(req) if not req.dry else {"ok": True}
        if not dres.get("ok"):
            st.state="failed"; _emit(f"auto_canary[{run_id}] deploy failed", priority=2); _status_set(req.pr_status,"failure","deploy failed"); return
        st.state="running"; _RUNS[run_id]=st

        while not _STOP.get(run_id):
            rr = readiness_ratio(req.namespace, req.app) if have_kubectl() else {"ratio": 1.0}
            st.last_ready_ratio=float(rr.get("ratio",0.0))

            # מדידות—prometheus או http
            if req.prom_url and (req.q_error_rate or req.q_latency_ms):
                err_series = _prom_series(req.prom_url, req.q_error_rate, req.prom_window_s) if req.q_error_rate else []
                lat_series = _prom_series(req.prom_url, req.q_latency_ms, req.prom_window_s) if req.q_latency_ms else []
                # spike detect
                err_spike = detect_spike(err_series) if err_series else {"spike":False}
                lat_spike = detect_lag_spike([x*1000.0 for x in lat_series]) if lat_series else {"spike":False}
                # ערכים אחרונים (fallback):
                m_err = err_series[-1] if err_series else 0.0
                m_lat = (lat_series[-1]*1000.0) if lat_series else 0.0
                st.last_error_rate = float(m_err)
                st.last_p95_ms      = float(m_lat)
                _emit(f"probe(prom) err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms ready={st.last_ready_ratio:.2f}")
                # Auto-rollback חכם על spike:
                if err_spike.get("spike") or lat_spike.get("spike"):
                    _emit("spike detected → rollback", priority=2)
                    canary_rollback({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                    st.state="rolled_back"
                    _status_set(req.pr_status, "failure", "canary rollback (spike)")
                    break
            elif req.probe_url:
                m = _http_probe(req.probe_url)
                st.last_error_rate=m["error_rate"]; st.last_p95_ms=m["p95_ms"]
                _emit(f"probe http err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms ready={st.last_ready_ratio:.2f}")
            else:
                st.last_error_rate=0.0; st.last_p95_ms=0.0
                _emit(f"probe none ready={st.last_ready_ratio:.2f}")

            ok = (st.last_error_rate <= pol.max_error_rate) and \
                 (st.last_p95_ms <= pol.max_p95_ms) and \
                 (st.last_ready_ratio >= pol.min_ready_ratio)

            if ok:
                st.ok_cycles += 1
                _emit(f"OK cycle {st.ok_cycles}/{pol.consecutive_ok_for_promote}")
                # promote Gate: Gatekeeper (אם נדרש)
                if st.ok_cycles >= pol.consecutive_ok_for_promote:
                    if req.gatekeeper_required and req.gate:
                        gate_dict=json.loads(req.gate.json())
                        gres = gate_evaluate(gate_dict)
                        record_gate_decision(run_id, "pre-promote", gate_dict, gres)
                        decision = "pass" if gres.get("ok") else f"fail:{gres.get('reasons')}"
                        BROKER.submit("timeline","auto-canary",{"type":"gatekeeper","ts":time.time(),"note":f"gatekeeper {decision}"}, priority=3)
                        if not gres.get("ok"):
                            st.state="gate_denied"
                            _status_set(req.pr_status, "failure", "gatekeeper denied")
                            if req.gate_fail_rollback:
                                canary_rollback({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                                _emit("rolled_back", priority=2); break
                            st.ok_cycles=0; time.sleep(pol.hold_seconds); continue
                    canary_promote({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                    st.state="promoted"; _emit("promoted", pct=100, priority=3)
                    _status_set(req.pr_status, "success", "canary promoted")
                    break
                # step
                if st.steps < pol.max_steps:
                    st.steps += 1
                    canary_step({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"add_percent":pol.step_percent,"total_replicas":req.total_replicas,"dry":req.dry})
            else:
                _emit(f"violation: err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms ready={st.last_ready_ratio:.2f}", priority=2)
                st.state="violated"
                _status_set(req.pr_status, "failure", "canary violation")
                if pol.rollback_on_first_violation:
                    canary_rollback({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                    st.state="rolled_back"; _emit("rolled_back", priority=2); break
                st.ok_cycles = 0

            for _ in range(pol.hold_seconds):
                if _STOP.get(run_id): break
                time.sleep(1.0)

        if _STOP.get(run_id): 
            st.state="stopped"
            _status_set(req.pr_status, "error", "canary stopped")

    except Exception as e:
        st.state=f"error:{e}"
        _status_set(req.pr_status, "error", "canary error")
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