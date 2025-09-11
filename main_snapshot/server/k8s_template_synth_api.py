# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict
from fastapi import APIRouter, Body
from fastapi.responses import PlainTextResponse

# תואם Pydantic v1 (כפי שהוצמד ב-Dockerfile)
try:
    from pydantic import BaseModel, Field
except Exception:  # אם מותקן v2 בתצורה אחרת
    from pydantic.v1 import BaseModel, Field  # type: ignore

router = APIRouter(prefix="/k8s", tags=["k8s-synth"])

class EnvVar(BaseModel):
    name: str = Field(..., min_length=1)
    value: str = ""

class Resources(BaseModel):
    limits: Optional[Dict[str, str]] = None
    requests: Optional[Dict[str, str]] = None

class DeploymentSpec(BaseModel):
    # פרמטרים בסיסיים
    name: str = Field(..., min_length=1)
    image: str = Field(..., min_length=1)
    replicas: int = 1
    port: int = 8000
    # אופציונלי
    env: Optional[List[EnvVar]] = None
    resources: Optional[Resources] = None
    # Service
    service: bool = True
    service_type: str = "ClusterIP"   # ClusterIP | NodePort | LoadBalancer
    service_port: int = 80

def render_deployment_yaml(spec: DeploymentSpec) -> str:
    """בניית YAML בלי שימוש ב-f-strings עם backslash בתוך הביטוי."""
    lines: List[str] = []
    # Deployment
    lines.append("apiVersion: apps/v1")
    lines.append("kind: Deployment")
    lines.append("metadata:")
    lines.append(f"  name: {spec.name}")
    lines.append("spec:")
    lines.append(f"  replicas: {spec.replicas}")
    lines.append("  selector:")
    lines.append("    matchLabels:")
    lines.append(f"      app: {spec.name}")
    lines.append("  template:")
    lines.append("    metadata:")
    lines.append("      labels:")
    lines.append(f"        app: {spec.name}")
    lines.append("    spec:")
    lines.append("      containers:")
    lines.append("        - name: app")
    lines.append(f"          image: {spec.image}")
    lines.append("          ports:")
    lines.append(f"            - containerPort: {spec.port}")

    # env (אם יש)
    if spec.env:
        lines.append("          env:")
        for ev in spec.env:
            safe_val = ev.value.replace('"', '\\"')
            lines.append(f"            - name: {ev.name}")
            lines.append(f'              value: "{safe_val}"')

    # resources (אם יש)
    if spec.resources and (spec.resources.limits or spec.resources.requests):
        lines.append("          resources:")
        if spec.resources.limits:
            lines.append("            limits:")
            for k, v in spec.resources.limits.items():
                lines.append(f"              {k}: {v}")
        if spec.resources.requests:
            lines.append("            requests:")
            for k, v in spec.resources.requests.items():
                lines.append(f"              {k}: {v}")

    result = "\n".join(lines)

    # Service (אופציונלי)
    if spec.service:
        svc: List[str] = []
        svc.append("---")
        svc.append("apiVersion: v1")
        svc.append("kind: Service")
        svc.append("metadata:")
        svc.append(f"  name: {spec.name}")
        svc.append("spec:")
        svc.append(f"  type: {spec.service_type}")
        svc.append("  selector:")
        svc.append(f"    app: {spec.name}")
        svc.append("  ports:")
        svc.append(f"    - port: {spec.service_port}")
        svc.append(f"      targetPort: {spec.port}")
        result = result + "\n" + "\n".join(svc)

    return result

@router.post("/template/synth", response_class=PlainTextResponse)
def synth_deployment(spec: DeploymentSpec = Body(...)) -> PlainTextResponse:
    """
    הפקת מניפסט K8s (Deployment + Service אופציונלי) לפי קלט JSON.
    החזרה היא טקסט YAML (text/plain) שניתן לשמור ישירות כקובץ.
    """
    yaml_text = render_deployment_yaml(spec)
    return PlainTextResponse(content=yaml_text, media_type="text/plain")
