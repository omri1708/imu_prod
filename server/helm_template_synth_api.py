# server/helm_template_synth_api.py
# Helm Chart generator: spec → chart under helm/generated/<slug>/ with Chart.yaml, values.yaml, templates/*.yaml
# Supports: Ingress / ServiceMonitor / NetworkPolicy (profiles) / IngressClass / cert-manager Issuer/Certificate
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
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

class IngressSpec(BaseModel):
    enabled: bool = False
    className: Optional[str] = None
    host: Optional[str] = None
    path: str = "/"
    tlsSecret: Optional[str] = None
    annotations: Dict[str,str] = {}

class IngressClassSpec(BaseModel):
    enabled: bool = False
    name: str = "nginx"
    controller: str = "k8s.io/ingress-nginx"

class ServiceMonitorSpec(BaseModel):
    enabled: bool = False
    scrapePort: int = 80
    interval: str = "30s"
    path: str = "/metrics"
    scheme: str = "http"
    labels: Dict[str,str] = {}

class NetworkPolicySpec(BaseModel):
    enabled: bool = False
    profile: str = Field("standard", regex="^(strict|standard|lenient)$")
    allowSameNamespace: bool = True
    ingressCidrs: list[str] = []
    egressCidrs: list[str] = []

class CertManagerSpec(BaseModel):
    enabled: bool = False
    issuerKind: str = Field("Issuer", regex="^(Issuer|ClusterIssuer)$")
    issuerName: str = "selfsigned"
    issuerNamespace: Optional[str] = None
    certificateSecretName: Optional[str] = None
    dnsNames: list[str] = []

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
    resources: Dict[str,str] = {}
    hpa: bool = False
    hpaMin: int = 2
    hpaMax: int = 10
    hpaCpu: int = 80
    ingress: IngressSpec = IngressSpec()
    ingressClass: IngressClassSpec = IngressClassSpec()
    serviceMonitor: ServiceMonitorSpec = ServiceMonitorSpec()
    networkPolicy: NetworkPolicySpec = NetworkPolicySpec()
    certManager: CertManagerSpec = CertManagerSpec()

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
    ingress = f"""
ingress:
  enabled: {str(s.ingress.enabled).lower()}
  className: {json.dumps(s.ingress.className) if s.ingress.className else 'null'}
  host: {json.dumps(s.ingress.host) if s.ingress.host else 'null'}
  path: {json.dumps(s.ingress.path)}
  tlsSecret: {json.dumps(s.ingress.tlsSecret) if s.ingress.tlsSecret else 'null'}
  annotations:
{''.join([f'    {k}: \"{v}\"\n' for k,v in s.ingress.annotations.items()]) if s.ingress.annotations else '    {}'}
"""
    ingress_class = f"""
ingressClass:
  enabled: {str(s.ingressClass.enabled).lower()}
  name: {json.dumps(s.ingressClass.name)}
  controller: {json.dumps(s.ingressClass.controller)}
"""
    sm = f"""
serviceMonitor:
  enabled: {str(s.serviceMonitor.enabled).lower()}
  scrapePort: {s.serviceMonitor.scrapePort}
  interval: "{s.serviceMonitor.interval}"
  path: "{s.serviceMonitor.path}"
  scheme: "{s.serviceMonitor.scheme}"
  labels:
{''.join([f'    {k}: \"{v}\"\n' for k,v in s.serviceMonitor.labels.items()]) if s.serviceMonitor.labels else '    {}'}
"""
    np = f"""
networkPolicy:
  enabled: {str(s.networkPolicy.enabled).lower()}
  profile: {json.dumps(s.networkPolicy.profile)}
  allowSameNamespace: {str(s.networkPolicy.allowSameNamespace).lower()}
  ingressCidrs:
{''.join([f'    - {cidr}\n' for cidr in s.networkPolicy.ingressCidrs]) if s.networkPolicy.ingressCidrs else '    []'}
  egressCidrs:
{''.join([f'    - {cidr}\n' for cidr in s.networkPolicy.egressCidrs]) if s.networkPolicy.egressCidrs else '    []'}
"""
    cm = f"""
certManager:
  enabled: {str(s.certManager.enabled).lower()}
  issuerKind: {json.dumps(s.certManager.issuerKind)}
  issuerName: {json.dumps(s.certManager.issuerName)}
  issuerNamespace: {json.dumps(s.certManager.issuerNamespace) if s.certManager.issuerNamespace else 'null'}
  certificateSecretName: {json.dumps(s.certManager.certificateSecretName) if s.certManager.certificateSecretName else 'null'}
  dnsNames:
{''.join([f'    - {json.dumps(n)}\n' for n in s.certManager.dnsNames]) if s.certManager.dnsNames else '    []'}
"""
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
{res}
hpa:
  enabled: {str(s.hpa).lower()}
  min: {s.hpaMin}
  max: {s.hpaMax}
  cpu: {s.hpaCpu}
{ingress_class}
{ingress}
{sm}
{np}
{cm}
"""

def tpl_helpers(name:str)->str:
    return f"""{{{{- define "{name}.name" -}}}}
{{{{ .Chart.Name }}}}
{{{{- end -}}}}

{{{{- define "{name}.fullname" -}}}}
{{{{ include "{name}.name" . }}}}-{{{{ .Release.Name }}}}
{{{{- end -}}}}
"""

def tpl_deployment(name:str)->str:
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{{{ include "{name}.fullname" . }}}}
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
spec:
  replicas: {{{{ .Values.replicaCount }}}}
  selector:
    matchLabels:
      app.kubernetes.io/name: {{{{ include "{name}.name" . }}}}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: {{{{ include "{name}.name" . }}}}
    spec:
      containers:
      - name: app
        image: "{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag }}}}"
        imagePullPolicy: "{{{{ .Values.image.pullPolicy }}}}"
        ports:
        - containerPort: {{{{ .Values.container.port }}}}
{{{{- if .Values.env }}}}
        env:
{{{{ toYaml .Values.env | indent 8 }}}}
{{{{- end }}}}
{{{{- if .Values.resources }}}}
{{{{ toYaml .Values.resources | indent 8 }}}}
{{{{- end }}}}
"""

def tpl_service(name:str)->str:
    return f"""apiVersion: v1
kind: Service
metadata:
  name: {{{{ include "{name}.fullname" . }}}}-svc
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
  labels:
    app.kubernetes.io/name: {{{{ include "{name}.name" . }}}}
spec:
  type: {{{{ .Values.service.type }}}}
  selector:
    app.kubernetes.io/name: {{{{ include "{name}.name" . }}}}
  ports:
  - name: http
    port: {{{{ .Values.service.port }}}}
    targetPort: {{{{ .Values.container.port }}}}
"""

def tpl_hpa(name:str)->str:
    return f"""{{{{- if .Values.hpa.enabled }}}}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{{{ include "{name}.fullname" . }}}}-hpa
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{{{ include "{name}.fullname" . }}}}
  minReplicas: {{{{ .Values.hpa.min }}}}
  maxReplicas: {{{{ .Values.hpa.max }}}}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{{{ .Values.hpa.cpu }}}}
{{{{- end }}}}
"""

def tpl_ingressclass(name:str)->str:
    return f"""{{{{- if .Values.ingressClass.enabled }}}}
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  name: {{{{ .Values.ingressClass.name }}}}
spec:
  controller: {{{{ .Values.ingressClass.controller }}}}
{{{{- end }}}}
"""

def tpl_ingress(name:str)->str:
    return f"""{{{{- if .Values.ingress.enabled }}}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{{{ include "{name}.fullname" . }}}}-ing
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
  annotations:
{{{{- if .Values.ingress.annotations }}}}
{{{{ toYaml .Values.ingress.annotations | indent 4 }}}}
{{{{- else }}}}
    {{}}
{{{{- end }}}}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{{{ .Values.ingress.className }}}}
  {{- end }}
  rules:
  - host: {{{{ .Values.ingress.host }}}}
    http:
      paths:
      - path: {{{{ .Values.ingress.path }}}}
        pathType: Prefix
        backend:
          service:
            name: {{{{ include "{name}.fullname" . }}}}-svc
            port:
              number: {{{{ .Values.service.port }}}}
  {{- if .Values.ingress.tlsSecret }}
  tls:
  - hosts:
    - {{{{ .Values.ingress.host }}}}
    secretName: {{{{ .Values.ingress.tlsSecret }}}}
  {{- end }}
{{{{- end }}}}
"""

def tpl_service_monitor(name:str)->str:
    return f"""{{{{- if .Values.serviceMonitor.enabled }}}}
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{{{ include "{name}.fullname" . }}}}-sm
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
  labels:
{{{{ toYaml .Values.serviceMonitor.labels | indent 4 }}}}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: {{{{ include "{name}.name" . }}}}
  endpoints:
  - port: http
    interval: {{{{ .Values.serviceMonitor.interval }}}}
    path: {{{{ .Values.serviceMonitor.path }}}}
    scheme: {{{{ .Values.serviceMonitor.scheme }}}}
{{{{- end }}}}
"""

def tpl_network_policy(name:str)->str:
    # מצבי profile: strict/standard/lenient
    return f"""{{{{- if .Values.networkPolicy.enabled }}}}
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{{{ include "{name}.fullname" . }}}}-np
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: {{{{ include "{name}.name" . }}}}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - {{}} # default drop depends on cluster policy; add rules below by profile
  {{- if eq .Values.networkPolicy.profile "strict" }}
  - from:
    {{- if .Values.networkPolicy.allowSameNamespace }}
    - podSelector: {{}}
    {{- end }}
    {{- range .Values.networkPolicy.ingressCidrs }}
    - ipBlock: {{ cidr: {{{{ . }}}} }}
    {{- end }}
  {{- else if eq .Values.networkPolicy.profile "standard" }}
  - from:
    - podSelector: {{}}
  {{- else }} # lenient
  - {{}} # allow all
  {{- end }}
  egress:
  - {{}} # base rule
  {{- if eq .Values.networkPolicy.profile "strict" }}
  - to:
    {{- range .Values.networkPolicy.egressCidrs }}
    - ipBlock: {{ cidr: {{{{ . }}}} }}
    {{- end }}
  {{- else if eq .Values.networkPolicy.profile "standard" }}
  - {{}} # allow cluster DNS/metadata left to cluster defaults
  {{- else }}
  - {{}} # allow all
  {{- end }}
{{{{- end }}}}
"""

def tpl_cert_issuer(name:str)->str:
    return f"""{{{{- if and .Values.certManager.enabled (eq .Values.certManager.issuerKind "Issuer") }}}}
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: {{{{ .Values.certManager.issuerName }}}}
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
spec:
  selfSigned: {{}}
{{{{- end }}}}
{{{{- if and .Values.certManager.enabled (eq .Values.certManager.issuerKind "ClusterIssuer") }}}}
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: {{{{ .Values.certManager.issuerName }}}}
spec:
  selfSigned: {{}}
{{{{- end }}}}
"""

def tpl_cert_certificate(name:str)->str:
    return f"""{{{{- if .Values.certManager.enabled }}}}
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: {{{{ include "{name}.fullname" . }}}}-crt
  namespace: {{{{ .Values.namespace | default .Release.Namespace }}}}
spec:
  secretName: {{{{ .Values.certManager.certificateSecretName | default (printf "%s-crt" (include "{name}.fullname" .)) }}}}
  issuerRef:
    name: {{{{ .Values.certManager.issuerName }}}}
    kind: {{{{ .Values.certManager.issuerKind }}}}
  dnsNames:
  {{- range .Values.certManager.dnsNames }}
  - {{{{ . }}}}
  {{- end }}
{{{{- end }}}}
"""

@router.post("/create")
def create(spec: ChartSpec):
    slug=_slug(spec.name)
    base=ROOT/slug; (base/"templates").mkdir(parents=True, exist_ok=True)
    (base/"Chart.yaml").write_text(chart_yaml(spec), encoding="utf-8")
    (base/"values.yaml").write_text(values_yaml(spec), encoding="utf-8")
    (base/"templates"/"_helpers.tpl").write_text(tpl_helpers(spec.name), encoding="utf-8")
    (base/"templates"/"deployment.yaml").write_text(tpl_deployment(spec.name), encoding="utf-8")
    (base/"templates"/"service.yaml").write_text(tpl_service(spec.name), encoding="utf-8")
    (base/"templates"/"hpa.yaml").write_text(tpl_hpa(spec.name), encoding="utf-8")
    (base/"templates"/"ingressclass.yaml").write_text(tpl_ingressclass(spec.name), encoding="utf-8")
    (base/"templates"/"ingress.yaml").write_text(tpl_ingress(spec.name), encoding="utf-8")
    (base/"templates"/"servicemonitor.yaml").write_text(tpl_service_monitor(spec.name), encoding="utf-8")
    (base/"templates"/"networkpolicy.yaml").write_text(tpl_network_policy(spec.name), encoding="utf-8")
    (base/"templates"/"issuer.yaml").write_text(tpl_cert_issuer(spec.name), encoding="utf-8")
    (base/"templates"/"certificate.yaml").write_text(tpl_cert_certificate(spec.name), encoding="utf-8")
    (base/"contract.json").write_text(json.dumps({"title":"Values","type":"object"}, indent=2), encoding="utf-8")
    (base/"README.md").write_text(f"# {spec.name} (Helm chart, auto-generated)\nWith Ingress/IngressClass/ServiceMonitor/NetworkPolicy/Certificate.\n", encoding="utf-8")
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
    return {"ok": bool(j.get("ok")), "reason": j.get("reason"), "cmd": j.get("cmd")}

@router.get("/get")
def get_template(slug: str):
    base=ROOT/slug
    if not base.exists(): raise HTTPException(404,"not found")
    files={}
    for n in ("Chart.yaml","values.yaml",
              "templates/_helpers.tpl","templates/deployment.yaml","templates/service.yaml",
              "templates/hpa.yaml","templates/ingressclass.yaml","templates/ingress.yaml",
              "templates/servicemonitor.yaml","templates/networkpolicy.yaml",
              "templates/issuer.yaml","templates/certificate.yaml",
              "contract.json","README.md"):
        p=base/n
        if p.exists(): files[str(n)] = p.read_text(encoding="utf-8")
    return {"ok": True, "files": files}