# server/runbook_api.py
# Runbook Orchestrator: תסריטי end-to-end מחוברים ל-WFQ + p95 Gate.
from __future__ import annotations
import os, platform
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time, urllib.request, json, shutil, asyncio
from policy.rbac import require_perm
from server.stream_wfq import BROKER
from runtime.p95 import GATES

router = APIRouter(prefix="/runbook", tags=["runbook"])

API = "http://127.0.0.1:8000"  # assume co-located; אם אחרת, ספקו ב-env

def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(API+path, method="POST", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _have(x:str)->bool: return shutil.which(x) is not None

class UnityK8sReq(BaseModel):

    user_id: str = "demo-user"
    project_dir: str
    target: str = "Android"
    namespace: str = "default"
    name: str = "unity-app"

@router.post("/unity_k8s")
def unity_k8s(req: UnityK8sReq):
    require_perm(req.user_id, "runbook:unity_k8s")
    t0=time.time()
    topic="timeline"
    BROKER.ensure_topic(topic, rate=100.0, burst=500, weight=2)
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":"runbook.unity_k8s.start"},priority=2)
    # 1) Unity
    unity_params={"project":req.project_dir,"target":req.target,"method":"Builder.PerformBuild","version":"2022.3.44f1","log":"/tmp/unity.log"}
    dry1=_post("/adapters/dry_run", {"user_id":req.user_id,"kind":"unity.build","params":unity_params})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":f"unity.dry.cmd={dry1.get('cmd')}"},priority=4)
    exec1=_have("unity") or _have("Unity") or _have("unity-editor")
    r1=_post("/adapters/run", {"user_id":req.user_id,"kind":"unity.build","params":unity_params,"execute":bool(exec1)})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":"unity.exec" if r1["ok"] else "unity.dry"},priority=4)
    # 2) K8s
    manifest=f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {req.name}, namespace: {req.namespace}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {req.name}}}}}
  template:
    metadata: {{labels: {{app: {req.name}}}}}
    spec:
      containers:
      - name: web
        image: nginx:alpine
"""
    dry2=_post("/adapters/dry_run", {"user_id":req.user_id,"kind":"k8s.kubectl.apply","params":{"manifest":manifest,"namespace":req.namespace}})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":f"k8s.dry.cmd={dry2.get('cmd')}"},priority=3)
    exec2=_have("kubectl")
    r2=_post("/adapters/run", {"user_id":req.user_id,"kind":"k8s.kubectl.apply","params":{"manifest":manifest,"namespace":req.namespace},"execute":bool(exec2)})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":"k8s.exec" if r2["ok"] else "k8s.dry"},priority=3)
    ms=(time.time()-t0)*1000
    GATES.observe("runbook.unity_k8s", ms)
    return {"ok": r1["ok"] and r2["ok"], "ms": ms, "unity": r1, "k8s": r2}

class AndroidReq(BaseModel):
    user_id: str = "demo-user"
    app_dir: str

@router.post("/android")
def android(rb: AndroidReq):
    require_perm(rb.user_id, "android")
    t0=time.time(); topic="timeline"; BROKER.ensure_topic(topic, rate=100.0, burst=500, weight=2)
    params={"flavor":"Release","buildType":"Aab","keystore":rb.app_dir+"/keystore.jks"}
    d=_post("/adapters/dry_run", {"user_id":rb.user_id,"kind":"android.gradle","params":params})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":f"android.dry.cmd={d.get('cmd')}"},priority=4)
    exec_ok = _have("gradle") or os.path.exists(os.path.join(rb.app_dir,"gradlew"))
    r=_post("/adapters/run", {"user_id":rb.user_id,"kind":"android.gradle","params":params,"execute":bool(exec_ok)})
    ms=(time.time()-t0)*1000
    GATES.observe("runbook.android", ms)
    return {"ok": r["ok"], "ms": ms, "android": r}

class IOSReq(BaseModel):
    user_id: str = "demo-user"
    workspace: str
    scheme: str = "App"
    config: str = "Release"

@router.post("/ios")
def ios(rb: IOSReq):
    require_perm(rb.user_id, "ios")
    t0=time.time(); topic="timeline"; BROKER.ensure_topic(topic, rate=100.0, burst=500, weight=2)
    params={"workspace":rb.workspace,"scheme":rb.scheme,"config":rb.config}
    d=_post("/adapters/dry_run", {"user_id":rb.user_id,"kind":"ios.xcode","params":params})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":f"ios.dry.cmd={d.get('cmd')}"},priority=4)
    exec_ok = (platform.system().lower()=="darwin") and shutil.which("xcodebuild")
    r=_post("/adapters/run", {"user_id":rb.user_id,"kind":"ios.xcode","params":params,"execute":bool(exec_ok)})
    ms=(time.time()-t0)*1000
    GATES.observe("runbook.ios", ms)
    return {"ok": r["ok"], "ms": ms, "ios": r}

class CUDAReq(BaseModel):
    user_id: str = "demo-user"
    src: str = "kern.cu"
    out: str = "kern"

@router.post("/cuda")
def cuda(rb: CUDAReq):
    require_perm(rb.user_id, "cuda")
    t0=time.time(); topic="timeline"; BROKER.ensure_topic(topic, rate=100.0, burst=500, weight=2)
    params={"src":rb.src,"out":rb.out}
    d=_post("/adapters/dry_run", {"user_id":rb.user_id,"kind":"cuda.nvcc","params":params})
    BROKER.submit(topic,"runbook",{"type":"event","ts":time.time(),"note":f"cuda.dry.cmd={d.get('cmd')}"},priority=4)
    exec_ok = shutil.which("nvcc") is not None
    r=_post("/adapters/run", {"user_id":rb.user_id,"kind":"cuda.nvcc","params":params,"execute":bool(exec_ok)})
    ms=(time.time()-t0)*1000
    GATES.observe("runbook.cuda", ms)
    return {"ok": r["ok"], "ms": ms, "cuda": r}


