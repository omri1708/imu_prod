# server/k8s_template_synth_api.py
# Spec → generate k8s templates under k8s/generated/<slug>/ (deployment.yaml, service.yaml, hpa.yaml?)
# + contract.json + README + API: list/get/dry_apply/apply
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time, re

from server.scheduler_api import _http_call  # reuse to call /adapters endpoints

router = APIRouter(prefix="/k8s/synth", tags=["k8s-synth"])

ROOT = Path("k8s/generated")
ROOT.mkdir(parents=True, exist_ok=True)

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+","-", s.lower()).strip("-")

class K8sContainer(BaseModel):
    name: str = "app"
    image: str
    port: int = 80
    env: Dict[str, str] = {}
    cpu: str | None = None
    memory: str | None = None

class K8sSpec(BaseModel):
    name: str
    namespace: str = "default"
    labels: Dict[str,str] = {}
    replicas: int = 2
    service_type: str = Field("ClusterIP", regex="^(ClusterIP|NodePort|LoadBalancer)$")
    container: K8sContainer
    hpa: bool = False
    hpa_min: int = 2
    hpa_max: int = 10
    hpa_cpu: int = 80  # target CPU %

def _deployment_yaml(s: K8sSpec) -> str:
    env_lines = "\n".join([f"            - name: {k}\n              value: \"{v}\"" for k,v in (s.container.env or {}).items()])
    res_lines = ""
    if s.container.cpu or s.container.memory:
        reqs=[]
        if s.container.cpu: reqs.append(f"              cpu: {s.container.cpu}")
        if s.container.memory: reqs.append(f"              memory: {s.container.memory}")
        res_lines = "          resources:\n            requests:\n" + "\n".join(reqs) + "\n"
    labels = "\n".join([f"      {k}: {v}" for k,v in s.labels.items()])
    if labels: labels="\n"+labels
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {s.name}
  namespace: {s.namespace}
  labels:{labels if labels else ""}
spec:
  replicas: {s.replicas}
  selector:
    matchLabels:
      app: {s.name}
  template:
    metadata:
      labels:
        app: {s.name}{labels if labels else ""}
    spec:
      containers:
      - name: {s.container.name}
        image: {s.container.image}
        ports:
        - containerPort: {s.container.port}
{('        env:\n'+env_lines) if env_lines else ''}{res_lines}"""
def _service_yaml(s: K8sSpec) -> str:
    labels = "\n".join([f"    {k}: {v}" for k,v in s.labels.items()])
    if labels: labels="\n"+labels
    return f"""apiVersion: v1
kind: Service
metadata:
  name: {s.name}-svc
  namespace: {s.namespace}
  labels:{labels if labels else ""}
spec:
  type: {s.service_type}
  selector:
    app: {s.name}
  ports:
  - port: {s.container.port}
    targetPort: {s.container.port}
"""
def _hpa_yaml(s: K8sSpec) -> str:
    return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {s.name}-hpa
  namespace: {s.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {s.name}
  minReplicas: {s.hpa_min}
  maxReplicas: {s.hpa_max}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {s.hpa_cpu}
"""

def _contract(s: K8sSpec) -> Dict[str,Any]:
    return {
      "$schema":"http://json-schema.org/draft-07/schema#",
      "title":"K8sTemplate",
      "type":"object",
      "required":["name","namespace","replicas","service_type","container"],
      "properties":{
        "name":{"type":"string","minLength":1},
        "namespace":{"type":"string","minLength":1},
        "replicas":{"type":"integer","minimum":1},
        "service_type":{"type":"string","enum":["ClusterIP","NodePort","LoadBalancer"]},
        "container":{
          "type":"object","required":["image","port"],
          "properties":{
            "name":{"type":"string"},
            "image":{"type":"string"},
            "port":{"type":"integer","minimum":1,"maximum":65535},
            "env":{"type":"object"},
            "cpu":{"type":"string"},
            "memory":{"type":"string"}
          }
        }
      },
      "additionalProperties": False
    }

@router.post("/create")
def create(spec: K8sSpec):
    slug=_slug(spec.name)
    base=ROOT/slug; base.mkdir(parents=True, exist_ok=True)
    # write files
    (base/"deployment.yaml").write_text(_deployment_yaml(spec), encoding="utf-8")
    (base/"service.yaml").write_text(_service_yaml(spec), encoding="utf-8")
    if spec.hpa:
        (base/"hpa.yaml").write_text(_hpa_yaml(spec), encoding="utf-8")
    (base/"contract.json").write_text(json.dumps(_contract(spec), ensure_ascii=False, indent=2), encoding="utf-8")
    (base/"README.md").write_text(f"# {spec.name}\n\nAuto-generated K8s template.\n", encoding="utf-8")
    return {"ok": True, "slug": slug, "dir": str(base)}

@router.get("/list")
def list_templates():
    items=[]
    for d in ROOT.glob("*/deployment.yaml"):
        slug=d.parent.name
        items.append({"slug":slug, "dir": str(d.parent)})
    return {"ok": True, "items": items}

@router.get("/get")
def get_template(slug: str):
    base=ROOT/slug
    if not base.exists(): raise HTTPException(404,"not found")
    files={}
    for n in ("deployment.yaml","service.yaml","hpa.yaml","contract.json","README.md"):
        p=base/n
        if p.exists(): files[n]=p.read_text(encoding="utf-8")
    return {"ok": True, "files": files}

class ApplyReq(BaseModel):
    slug: str
    user_id: str = "demo-user"
    namespace: Optional[str] = None
    dry_run: bool = True

@router.post("/dry_apply")
def dry_apply(req: ApplyReq):
    base=ROOT/req.slug
    if not base.exists(): raise HTTPException(404,"not found")
    applied=[]
    for n in ("deployment.yaml","service.yaml","hpa.yaml"):
        p=base/n
        if not p.exists(): continue
        manifest=p.read_text(encoding="utf-8")
        body={"user_id":req.user_id,"kind":"k8s.kubectl.apply","params":{"manifest":manifest,"namespace": req.namespace or "default"}}
        j=_http_call("POST","/adapters/dry_run", body)
        applied.append({"file":n,"ok":j.get("ok",False),"cmd":j.get("cmd","")})
    return {"ok": True, "applied": applied}

@router.post("/apply")
def apply(req: ApplyReq):
    base=ROOT/req.slug
    if not base.exists(): raise HTTPException(404,"not found")
    applied=[]
    for n in ("deployment.yaml","service.yaml","hpa.yaml"):
        p=base/n
        if not p.exists(): continue
        manifest=p.read_text(encoding="utf-8")
        body={"user_id":req.user_id,"kind":"k8s.kubectl.apply","params":{"manifest":manifest,"namespace": req.namespace or "default"},"execute": True}
        j=_http_call("POST","/adapters/run", body)
        applied.append({"file":n,"ok":j.get("ok",False),"reason":j.get("reason","")})
    return {"ok": True, "applied": applied}