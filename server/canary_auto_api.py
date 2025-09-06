# server/canary_auto_api.py (UPDATED)
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time, threading, json, urllib.request, urllib.error, shutil

from server.canary_controller import deploy as canary_deploy, step as canary_step, promote as canary_promote, rollback as canary_rollback
from server.stream_wfq import BROKER
from policy.rbac import require_perm
from .canary_auto_policy import AutoCanaryPolicy
from .k8s_ready import readiness_ratio, have_kubectl
from .prometheus_client import query_instant, query_range, extract_last_vector, quantile_from_range

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
    # מקור מדידה:
    probe_url: Optional[str] = None          # אם קיים – יבצע HTTP probes
    prom_url: Optional[str] = None           # אם קיים – ישתמש ב-Prometheus
    q_error_rate: Optional[str] = None       # למשל rate(http_requests_total{status=~"5.."}[3m]) / rate(http_requests_total[3m])
    q_latency_ms: Optional[str] = None       # למשל histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[3m])) by (le))
    prom_window_s: int = 180
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

def _prom_eval(prom_url: str, q_err: str, q_lat: str, window_s: int) -> Dict[str,Any]:
    now=time.time()
    er = extract_last_vector(query_instant(prom_url, q_err, ts=now)) if q_err else None
    lat=None
    if q_lat:
        rng = query_range(prom_url, q_lat, start=now-window_s, end=now, step=max(5, window_s//30))
        v = quantile_from_range(rng, 0.95)
        lat = v*1000.0 if v is not None else None  # שניות→מילישניות
    return {"error_rate": float(er) if er is not None else 0.0, "p95_ms": float(lat) if lat is not None else 0.0}

def _loop(run_id: str, req: StartReq):
    pol = req.policy
    st = Status(run_id=run_id, state="starting", ok_cycles=0, steps=0,
                last_error_rate=1.0, last_p95_ms=9999.0, last_ready_ratio=0.0, started_ts=time.time())
    _RUNS[run_id]=st
    try:
        # initial deploy
        _emit(f"auto_canary[{run_id}] deploy baseline/canary {req.total_replicas}/{req.canary_percent}%", pct=5)
        dres = canary_deploy(req) if not req.dry else {"ok": True}
        if not dres.get("ok"):
            st.state="failed"; _emit(f"auto_canary[{run_id}] deploy failed", priority=2); return
        st.state="running"; _RUNS[run_id]=st

        # loop
        while not _STOP.get(run_id):
            # readiness from k8s (if kubectl available)
            rr = readiness_ratio(req.namespace, req.app) if have_kubectl() else {"ratio": 1.0}
            st.last_ready_ratio=float(rr.get("ratio",0.0))
            # measures: prometheus or http probe
            if req.prom_url and (req.q_error_rate or req.q_latency_ms):
                m = _prom_eval(req.prom_url, req.q_error_rate, req.q_latency_ms, req.prom_window_s)
            elif req.probe_url:
                m = _http_probe(req.probe_url)
            else:
                m = {"error_rate":0.0,"p95_ms":0.0}
            st.last_error_rate=m["error_rate"]; st.last_p95_ms=m["p95_ms"]; _RUNS[run_id]=st
            _emit(f"probe err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms ready={st.last_ready_ratio:.2f}")

            ok = (st.last_error_rate <= pol.max_error_rate) and \
                 (st.last_p95_ms <= pol.max_p95_ms) and \
                 (st.last_ready_ratio >= pol.min_ready_ratio)
            if ok:
                st.ok_cycles += 1
                _emit(f"OK cycle {st.ok_cycles}/{pol.consecutive_ok_for_promote}")
                if st.ok_cycles >= pol.consecutive_ok_for_promote:
                    canary_promote({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                    st.state="promoted"; _emit("promoted", pct=100, priority=3); break
                if st.steps < pol.max_steps:
                    st.steps += 1
                    canary_step({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"add_percent":pol.step_percent,"total_replicas":req.total_replicas,"dry":req.dry})
            else:
                _emit(f"violation: err={st.last_error_rate:.3f} p95={st.last_p95_ms:.0f}ms ready={st.last_ready_ratio:.2f}", priority=2)
                st.state="violated"
                if pol.rollback_on_first_violation:
                    canary_rollback({"user_id":req.user_id,"namespace":req.namespace,"app":req.app,"total_replicas":req.total_replicas,"dry":req.dry})
                    st.state="rolled_back"; _emit("rolled_back", priority=2); break
                st.ok_cycles = 0
            # sleep
            for _ in range(pol.hold_seconds):
                if _STOP.get(run_id): break
                time.sleep(1.0)
        if _STOP.get(run_id): st.state="stopped"
    except Exception as e:
        st.state=f"error:{e}"
    finally:
        st.stopped_ts=time.time(); _RUNS[run_id]=st; _STOP.pop(run_id, None)