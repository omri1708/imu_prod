# server/canary_controller.py
# Canary Controller: יצירה/שלבים/רולבק/פרומוט בעזרת kubectl.
# משקל ע"י חלוקת replicas בין baseline/canary תחת אותו Service (selector משותף).
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import subprocess, shutil, json, time

from server.stream_wfq import BROKER
from policy.rbac import require_perm

router = APIRouter(prefix="/canary", tags=["canary"])

def have(x:str)->bool: return shutil.which(x) is not None

def _kubectl(args: list[str], input_str: Optional[str]=None) -> str:
    p = subprocess.run(["kubectl"]+args, input=input_str, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout or p.stderr)
    return p.stdout

def _emit(note:str, pct: Optional[float]=None, priority:int=4, topic:str="timeline"):
    BROKER.ensure_topic(topic, rate=100, burst=500, weight=2)
    e={"type": "progress" if pct is not None else "event", "ts": time.time(), "note": note}
    if pct is not None: e["pct"]=pct
    BROKER.submit(topic,"canary",e, priority=priority)

class CanaryPlan(BaseModel):
    user_id: str = "demo-user"
    namespace: str = "default"
    app: str = "imu-app"
    image: str
    total_replicas: int = Field(ge=1, default=10)
    canary_percent: int = Field(ge=0, le=100, default=10)
    port: int = 80
    dry: bool = False

def _manifest(app:str, image:str, ns:str, replicas:int, labels:dict) -> dict:
    return {
      "apiVersion":"apps/v1","kind":"Deployment",
      "metadata":{"name":app,"namespace":ns,"labels":labels},
      "spec":{"replicas":replicas,
              "selector":{"matchLabels":{"app":app, **labels}},
              "template":{"metadata":{"labels":{"app":app, **labels}},
                          "spec":{"containers":[{"name":"web","image":image,"ports":[{"containerPort":80}],
                                                 "readinessProbe":{"httpGet":{"path":"/","port":80},"initialDelaySeconds":3,"periodSeconds":5}}]}}}
    }

def _service(app:str, ns:str, port:int) -> dict:
    return {"apiVersion":"v1","kind":"Service",
            "metadata":{"name":f"{app}-svc","namespace":ns},
            "spec":{"selector":{"app":app}, "ports":[{"port":port,"targetPort":80}]}}

@router.post("/deploy")
def deploy(plan: CanaryPlan):
    require_perm(plan.user_id, "canary:deploy")
    if not have("kubectl"):
        return {"ok": False, "resource_required": "kubectl", "install": "brew install kubectl | winget install -e --id Kubernetes.kubectl"}
    # baseline + canary replicas by percent
    canary = max(0, min(plan.total_replicas, plan.total_replicas * plan.canary_percent // 100))
    baseline = plan.total_replicas - canary
    if baseline==0 and canary==0: baseline=1
    base_labels={"track":"baseline"}
    can_labels={"track":"canary"}
    base_dep=_manifest(plan.app, plan.image, plan.namespace, baseline, base_labels)
    can_dep=_manifest(plan.app, plan.image, plan.namespace, canary, can_labels)
    svc=_service(plan.app, plan.namespace, plan.port)

    _emit(f"canary.deploy start app={plan.app} {baseline}/{canary}", pct=5)
    if not plan.dry:
        _kubectl(["apply","-f","-"], json.dumps(base_dep))
        if canary>0:
            _kubectl(["apply","-f","-"], json.dumps(can_dep))
        _kubectl(["apply","-f","-"], json.dumps(svc))
    _emit("canary.deploy done", pct=40)
    return {"ok": True, "baseline": baseline, "canary": canary}

class StepReq(BaseModel):
    user_id: str = "demo-user"
    namespace: str = "default"
    app: str
    add_percent: int = Field(ge=1, le=100)
    total_replicas: int = Field(ge=1, default=10)
    dry: bool = False

@router.post("/step")
def step(req: StepReq):
    require_perm(req.user_id, "canary:step")
    if not have("kubectl"):
        return {"ok": False, "resource_required": "kubectl"}
    # get current replica distribution
    out = _kubectl(["get","deploy","-n",req.namespace,"-l",f"app={req.app}", "-o","json"])
    j = json.loads(out)
    base=can=0
    for i in j.get("items",[]):
        if i["metadata"]["labels"].get("track")=="baseline":
            base=i["spec"]["replicas"]
        elif i["metadata"]["labels"].get("track")=="canary":
            can=i["spec"]["replicas"]
    # new canary count
    delta = max(1, req.total_replicas * req.add_percent // 100)
    new_can = min(req.total_replicas, can + delta)
    new_base = max(0, req.total_replicas - new_can)
    _emit(f"canary.step {can}->{new_can}", pct=70)
    if not req.dry:
        _kubectl(["scale","deploy","-n",req.namespace,"-l",f"app={req.app},track=baseline", f"--replicas={new_base}"])
        _kubectl(["scale","deploy","-n",req.namespace,"-l",f"app={req.app},track=canary", f"--replicas={new_can}"])
    return {"ok": True, "canary": new_can, "baseline": new_base}

class PromoteReq(BaseModel):
    user_id: str = "demo-user"
    namespace: str = "default"
    app: str
    total_replicas: int = 10
    dry: bool = False

@router.post("/promote")
def promote(req: PromoteReq):
    require_perm(req.user_id, "canary:promote")
    if not have("kubectl"):
        return {"ok": False, "resource_required": "kubectl"}
    _emit("canary.promote", pct=90)
    if not req.dry:
        # scale canary to total; baseline to 0
        _kubectl(["scale","deploy","-n",req.namespace,"-l",f"app={req.app},track=baseline", f"--replicas=0"])
        _kubectl(["scale","deploy","-n",req.namespace,"-l",f"app={req.app},track=canary", f"--replicas={req.total_replicas}"])
    return {"ok": True}

class RollbackReq(BaseModel):
    user_id: str = "demo-user"
    namespace: str = "default"
    app: str
    total_replicas: int = 10
    dry: bool = False

@router.post("/rollback")
def rollback(req: RollbackReq):
    require_perm(req.user_id, "canary:rollback")
    if not have("kubectl"):
        return {"ok": False, "resource_required": "kubectl"}
    _emit("canary.rollback", pct=95)
    if not req.dry:
        _kubectl(["scale","deploy","-n",req.namespace,"-l",f"app={req.app},track=canary", f"--replicas=0"])
        _kubectl(["scale","deploy","-n",req.namespace,"-l",f"app={req.app},track=baseline", f"--replicas={req.total_replicas}"])
    return {"ok": True}