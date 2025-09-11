# engine/contracts_policy.py
from __future__ import annotations
from typing import Dict, Any
import os, json, time
from adapters.contracts.base import ResourceRequired, ProcessFailed, ContractError, record_event

DEFAULT_POLICY = {
  "trust": "standard",            # "low"/"standard"/"high"
  "ttl_days": 90,                 # retention
  "quotas": {"builds_per_hour": 30, "jobs_per_hour": 60},
  "topics": {"timeline":{"rate":20,"burst":80}, "progress":{"rate":50,"burst":200}, "logs":{"rate":200,"burst":400}}
}

def _load_user_policy() -> Dict[str, Any]:
    p=os.environ.get("IMU_USER_POLICY_JSON")
    if not p: return DEFAULT_POLICY
    try: return json.loads(p)
    except: return DEFAULT_POLICY

_RATE_BUCKETS: Dict[str, list[float]] = {"builds":[], "jobs":[]}

def _allow(counter: str, per_hour: int) -> bool:
    now=time.time(); window=3600.0
    xs=_RATE_BUCKETS.setdefault(counter,[])
    xs[:] = [t for t in xs if now-t < window]
    if len(xs) >= per_hour: return False
    xs.append(now); return True

def policy_wrap(op_name: str, fn, *args, **kwargs):
    pol=_load_user_policy()
    if "build" in op_name:
        if not _allow("builds", pol["quotas"]["builds_per_hour"]):
            return {"ok":False, "error":{"type":"quota","msg":"builds_per_hour exceeded"}}
    if op_name in ("k8s.job",):
        if not _allow("jobs", pol["quotas"]["jobs_per_hour"]):
            return {"ok":False, "error":{"type":"quota","msg":"jobs_per_hour exceeded"}}
    try:
        res=fn(*args, **kwargs)
        record_event("adapter.ok", {"op":op_name})
        return {"ok":True, "result": res}
    except ResourceRequired as rr:
        record_event("adapter.need", {"op":op_name,"need":rr.resource})
        return {"ok":False, "need":{"resource":rr.resource,"install":rr.how_to_install,"why":rr.why}}
    except ProcessFailed as pf:
        record_event("adapter.fail", {"op":op_name,"rc":pf.rc,"cmd":pf.cmd,"err":pf.err[-500:]})
        return {"ok":False,"error":{"type":"process_failed","rc":pf.rc}}
    except ContractError as ce:
        record_event("adapter.contract_error", {"op":op_name,"msg":str(ce)})
        return {"ok":False,"error":{"type":"contract_error","msg":str(ce)}}