# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict
from fastapi import APIRouter, Body
from fastapi.responses import PlainTextResponse

# תואם Pydantic v1 (כפי שהצמדנו בדוקר)
try:
    from pydantic import BaseModel, Field
except Exception:  # אם יש התקנה חלופית של v2
    from pydantic.v1 import BaseModel, Field  # type: ignore

router = APIRouter(prefix="/helm", tags=["helm-synth"])

class EnvVar(BaseModel):
    name: str = Field(..., min_length=1)
    value: str = ""

class Resources(BaseModel):
    limits: Optional[Dict[str, str]] = None
    requests: Optional[Dict[str, str]] = None

class HelmSpec(BaseModel):
    chart_name: str = Field(..., min_length=1)
    app_name: str = Field(..., min_length=1)
    image: str = Field(..., min_length=1)
    replicas: int = 1
    container_port: int = 8000
    env: Optional[List[EnvVar]] = None
    resources: Optional[Resources] = None
    service: bool = True
    service_type: str = "ClusterIP"   # ClusterIP | NodePort | LoadBalancer
    service_port: int = 80
    ingress: bool = False
    host: Optional[str] = None
    path: str = "/"

def _chart_yaml(spec: HelmSpec) -> str:
    lines: List[str] = []
    lines.append("apiVersion: v2")
    lines.append("name: " + spec.chart_name)
    lines.append("type: application")
    lines.append("version: 0.1.0")
    lines.append('appVersion: "1.0.0"')
    return "\n".join(lines)

def _values_yaml(spec: HelmSpec) -> str:
    lines: List[str] = []
    lines.append('nameOverride: ""')
    lines.append('fullnameOverride: ""')
    lines.append("")
    lines.append("image:")
    lines.append("  repository: " + spec.image)
    lines.append("  pullPolicy: IfNotPresent")
    lines.append("")
    lines.append("replicaCount: " + str(spec.replicas))
    lines.append("")
    lines.append("service:")
    lines.append("  type: " + spec.service_type)
    lines.append("  port: " + str(spec.service_port))
    lines.append("")
    lines.append("containerPort: " + str(spec.container_port))
    lines.append("")
    if spec.env:
        lines.append("env:")
        for ev in spec.env:
            safe_val = (ev.value or "").replace('"', '\\"')
            lines.append("  - name: " + ev.name)
            lines.append('    value: "' + safe_val + '"')
        lines.append("")
    if spec.resources and (spec.resources.limits or spec.resources.requests):
        lines.append("resources:")
        if spec.resources.limits:
            lines.append("  limits:")
            for k, v in spec.resources.limits.items():
                lines.append(f"    {k}: {v}")
        if spec.resources.requests:
            lines.append("  requests:")
            for k, v in spec.resources.requests.items():
                lines.append(f"    {k}: {v}")
        lines.append("")
    # ingress ערכים
    lines.append("ingress:")
    if spec.ingress and spec.host:
        lines.append("  enabled: true")
        lines.append('  className: ""')
        lines.append("  hosts:")
        lines.append("    - host: " + spec.host)
        lines.append("      paths:")
        lines.append("        - path: " + (spec.path or "/"))
        lines.append("          pathType: Prefix")
        lines.append("  tls: []")
    else:
        lines.append("  enabled: false")
    return "\n".join(lines)

def _deployment_template(spec: HelmSpec) -> str:
    # חשוב: פה אין שימוש ב-f-strings עם {{ }} כדי שלא יישבר פייתון; אלו מחרוזות רגילות.
    L = []
    L.append("apiVersion: apps/v1")
    L.append("kind: Deployment")
    L.append("metadata:")
    L.append("  name: {{ .Chart.Name }}")
    L.append("  labels:")
    L.append("    app.kubernetes.io/name: {{ .Chart.Name }}")
    L.append("    app.kubernetes.io/instance: {{ .Release.Name }}")
    L.append("spec:")
    L.append("  replicas: {{ .Values.replicaCount }}")
    L.append("  selector:")
    L.append("    matchLabels:")
    L.append("      app.kubernetes.io/name: {{ .Chart.Name }}")
    L.append("  template:")
    L.append("    metadata:")
    L.append("      labels:")
    L.append("        app.kubernetes.io/name: {{ .Chart.Name }}")
    L.append("        app.kubernetes.io/instance: {{ .Release.Name }}")
    L.append("    spec:")
    L.append("      containers:")
    L.append("        - name: " + spec.app_name)
    L.append('          image: "{{ .Values.image.repository }}"')
    L.append("          ports:")
    L.append("            - containerPort: {{ .Values.containerPort }}")
    # env
    L.append("          env:")
    L.append("          {{- if .Values.env }}")
    L.append("          {{- range .Values.env }}")
    L.append('            - name: {{ .name | quote }}')
    L.append('              value: {{ .value | quote }}')
    L.append("          {{- end }}")
    L.append("          {{- end }}")
    # resources
    L.append("          resources:")
    L.append("          {{- toYaml .Values.resources | nindent 10 }}")
    return "\n".join(L)

def _service_template() -> str:
    L = []
    L.append("apiVersion: v1")
    L.append("kind: Service")
    L.append("metadata:")
    L.append("  name: {{ .Chart.Name }}")
    L.append("spec:")
    L.append("  type: {{ .Values.service.type }}")
    L.append("  selector:")
    L.append("    app.kubernetes.io/name: {{ .Chart.Name }}")
    L.append("  ports:")
    L.append("    - port: {{ .Values.service.port }}")
    L.append("      targetPort: {{ .Values.containerPort }}")
    return "\n".join(L)

def _ingress_template() -> str:
    L = []
    L.append("apiVersion: networking.k8s.io/v1")
    L.append("kind: Ingress")
    L.append("metadata:")
    L.append("  name: {{ .Chart.Name }}")
    L.append("spec:")
    L.append("  rules:")
    L.append("    - host: {{ (index .Values.ingress.hosts 0).host | default \"\" }}")
    L.append("      http:")
    L.append("        paths:")
    L.append("          - path: {{ (index (index .Values.ingress.hosts 0).paths 0).path | default \"/\" }}")
    L.append("            pathType: Prefix")
    L.append("            backend:")
    L.append("              service:")
    L.append("                name: {{ .Chart.Name }}")
    L.append("                port:")
    L.append("                  number: {{ .Values.service.port }}")
    return "\n".join(L)

def render_helm_chart(spec: HelmSpec) -> str:
    """
    מחזיר טקסט אחד שמייצג את כל קבצי הצ'ארט, עם מפרידי קבצים.
    זה נוח להדבקה/שמירה; אפשר לפצל אחר כך לפי הכותרות.
    """
    parts: List[str] = []
    parts.append("# file: Chart.yaml\n" + _chart_yaml(spec))
    parts.append("# file: values.yaml\n" + _values_yaml(spec))
    parts.append("# file: templates/deployment.yaml\n" + _deployment_template(spec))
    parts.append("# file: templates/service.yaml\n" + _service_template())
    if spec.ingress and spec.host:
        parts.append("# file: templates/ingress.yaml\n" + _ingress_template())
    return "\n---\n".join(parts) + "\n"

@router.post("/template/synth", response_class=PlainTextResponse)
def synth_chart(spec: HelmSpec = Body(...)) -> PlainTextResponse:
    """
    הפקת Chart בסיסי של Helm (Chart.yaml / values.yaml / templates/*.yaml) מטופס JSON.
    החזרה היא טקסט (text/plain) שניתן לשמור כקבצים לפי הכותרות (# file: ...).
    """
    txt = render_helm_chart(spec)
    return PlainTextResponse(content=txt, media_type="text/plain")
