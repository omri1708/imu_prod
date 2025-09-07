# server/controlplane_deploy_api.py
# One-click deploy of control-plane chart via adapters.{helm.upgrade} (בכפוף לקיום helm)
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path

from server.scheduler_api import _http_call
from policy.rbac import require_perm

router = APIRouter(prefix="/controlplane", tags=["controlplane"])

class DeployReq(BaseModel):
  user_id: str = "demo-user"
  release: str = "imu"
  namespace: str = "default"
  values_file: Optional[str] = None
  execute: bool = True   # אם helm קיים — ירוץ; אחרת נקבל resource_required (וזה בסדר)

@router.post("/deploy")
def deploy(req: DeployReq):
  require_perm(req.user_id, "helm:deploy")
  chart_dir = str(Path("helm/control-plane").resolve())
  vf = req.values_file or str(Path(chart_dir, "values.yaml"))
  body = {
    "user_id": req.user_id,
    "kind": "helm.upgrade",
    "params": {
      "release": req.release,
      "chart_dir": chart_dir,
      "namespace": req.namespace,
      "values_file": vf,
      "extra_opt": ""
    },
    "execute": bool(req.execute)
  }
  out = _http_call("POST", "/adapters/run", body)
  return {"ok": out.get("ok", False), "reason": out.get("reason"), "cmd": out.get("cmd")}

class DryReq(BaseModel):
  release: str = "imu"
  namespace: str = "default"
  values_file: Optional[str] = None

@router.post("/dry")
def dry(req: DryReq):
  chart_dir = str(Path("helm/control-plane").resolve())
  vf = req.values_file or str(Path(chart_dir, "values.yaml"))
  body = {
    "user_id": "demo-user",
    "kind": "helm.upgrade",
    "params": {
      "release": req.release,
      "chart_dir": chart_dir,
      "namespace": req.namespace,
      "values_file": vf,
      "extra_opt": " --dry-run --debug"
    },
    "execute": False
  }
  out = _http_call("POST", "/adapters/run", body)
  return {"ok": out.get("ok", False), "cmd": out.get("cmd")}