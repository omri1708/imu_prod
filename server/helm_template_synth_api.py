# server/helm_template_synth_api.py
# Helm Chart generator: spec → chart under helm/generated/<slug>/ with Chart.yaml, values.yaml, templates/*.yaml
# API: create/list/get/dry_template/upgrade (dry/run)
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
import json, re

from server.scheduler_api import _http_call  # reuse HTTP helper

router = APIRouter(prefix="/helm/synth", tags=["helm-synth"])
ROOT = Path("helm/generated")
ROOT.mkdir(parents=True, exist_ok=True)

def _slug(s:str)->str:
    return re.sub(r"[^a-z0-9\-]+","-", s.lower()).strip("-")

class ChartImage(BaseModel):
    repository: str
    tag: str = "latest"
    pullPolicy: str = "IfNotPresent"

class ChartSpec(BaseModel):
    name: str
    version: str = "0.1.0"
    appVersion: str = "1.0.0"
    namespace: str = "default"
    release: str = "release"
    serviceType: str = Field("ClusterIP", regex="^(ClusterIP|NodePort|LoadBalancer)$")
    replicas: int = 2
    containerPort: int = 80
    image: ChartImage
    env: Dict[str,str] = {}
    resources: Dict[str,str] = {}   # {"cpu":"100m","memory":"128Mi"}
    hpa: bool = False
    hpaMin: int = 2
    hpaMax: int = 10
    hpaCpu: int = 80

def chart_yaml(s:ChartSpec)->str:
    return f"""apiVersion: v2
name: {s.name}
description: Auto-generated chart
type: application
version: {s.version}
appVersion: "{s.appVersion}"
"""

def values_yaml(s:ChartSpec)->str:
    env = "\n".join([f"  - name: {k}\n    value: \"{v}\"" for k,v in (s.env or {}).items()])
    res = ""
    if s.resources:
        res = "  resources:\n    requests:\n" + "".join([f"      {k}: {v}\n" for k,v in s.resources.items()])
    return f"""namespace: {s.namespace}
replicaCount: {s.replicas}
service:
  type: {s.serviceType}
  port: {s.containerPort}
image:
  repository: {s.image.repository}
  tag: {s.image.tag}
  pullPolicy: {s.image.pullPolicy}
container:
  port: {s.containerPort}
{('env:\n'+env) if env else ''}
{res}"""

def tpl_deployment()->str:
    return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "%s.fullname" . }}
  namespace: {{ .Values.namespace | default .Release.Namespace }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: {{ include "%s.name" . }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: {{ include "%s.name" . }}
    spec:
      containers:
      - name: app
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: "{{ .Values.image.pullPolicy }}"
        ports:
        - containerPort: {{ .Values.container.port }}
{{- if .Values.env }}
        env:
{{ toYaml .Values.env | indent 8 }}
{{- end }}
{{- if .Values.resources }}
{{ toYaml .Values.resources | indent 8 }}
{{- end }}
""" % ("%s","%s","%s")

def tpl_service()->str:
    return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "%s.fullname" . }}-svc
  namespace: {{ .Values.namespace | default .Release.Namespace }}
spec:
  type: {{ .Values.service.type }}
  selector:
    app.kubernetes.io/name: {{ include "%s.name" . }}
  ports:
  - port: {{ .Values.service.port }}
    targetPort: {{ .Values.container.port }}
""" % ("%s","%s")

def tpl_helpers()->str:
    return """{{- define "%s.name" -}}
%s
{{- end -}}

{{- define "%s.fullname" -}}
{{ include "%s.name" . }}-{{ .Release.Name }}
{{- end -}}
""" % ("%s","{{ .Chart.Name }}","%s","%s")

def tpl_hpa()->str:
    return """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "%s.fullname" . }}-hpa
  namespace: {{ .Values.namespace | default .Release.Namespace }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "%s.fullname" . }}
  minReplicas: {{ .Values.hpa.min }}
  maxReplicas: {{ .Values.hpa.max }}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{ .Values.hpa.cpu }}
""" % ("%s","%s")

@router.post("/create")
def create(spec: ChartSpec):
    slug=_slug(spec.name)
    base=ROOT/slug; (base/"templates").mkdir(parents=True, exist_ok=True)
    # files
    (base/"Chart.yaml").write_text(chart_yaml(spec), encoding="utf-8")
    vals = values_yaml(spec)+("\nhpa:\n  min: %d\n  max: %d\n  cpu: %d\n" % (spec.hpaMin,spec.hpaMax,spec.hpaCpu) if spec.hpa else "")
    (base/"values.yaml").write_text(vals, encoding="utf-8")
    (base/"templates"/"_helpers.tpl").write_text(tpl_helpers().replace("%s", spec.name), encoding="utf-8")
    (base/"templates"/"deployment.yaml").write_text(tpl_deployment().replace("%s", spec.name), encoding="utf-8")
    (base/"templates"/"service.yaml").write_text(tpl_service().replace("%s", spec.name), encoding="utf-8")
    if spec.hpa:
        (base/"templates"/"hpa.yaml").write_text(tpl_hpa().replace("%s", spec.name), encoding="utf-8")
    # minimal contract of values (exported for reference)
    (base/"contract.json").write_text(json.dumps({"title":"Values","type":"object"}, indent=2), encoding="utf-8")
    (base/"README.md").write_text(f"# {spec.name} (Helm chart)\n\nAuto-generated.\n", encoding="utf-8")
    return {"ok": True, "slug": slug, "dir": str(base), "release": spec.release, "namespace": spec.namespace}

@router.get("/list")
def list_charts():
    items=[]
    for d in ROOT.glob("*/Chart.yaml"):
        items.append({"slug": d.parent.name, "dir": str(d.parent)})
    return {"ok": True, "items": items}

@router.post("/dry_template")
def dry_template(slug: str, name: str, values_file: Optional[str] = None):
    base=ROOT/slug
    if not base.exists(): raise HTTPException(404,"not found")
    vf = values_file or str(base/"values.yaml")
    body={"user_id":"demo-user","kind":"helm.template","params":{"name": name, "chart_dir": str(base), "values_file": vf}}
    j=_http_call("POST","/adapters/dry_run", body)
    return {"ok": bool(j.get("ok")), "cmd": j.get("cmd","")}

@router.post("/upgrade")
def upgrade(slug: str, release: str, namespace: str = "default", values_file: Optional[str] = None, execute: bool = False):
    base=ROOT/slug
    if not base.exists(): raise HTTPException(404,"not found")
    vf = values_file or str(base/"values.yaml")
    body={"user_id":"demo-user","kind":"helm.upgrade","params":{"release":release,"chart_dir":str(base),"namespace":namespace,"values_file":vf,"extra_opt":""},"execute": execute}
    j=_http_call("POST","/adapters/run", body)
    return {"ok": bool(j.get("ok")),"reason": j.get("reason"),"cmd": j.get("cmd")}