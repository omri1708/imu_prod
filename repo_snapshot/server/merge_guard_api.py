# server/merge_guard_api.py
# GitOps Merge Guard: מריץ Gatekeeper (evidence + checks + p95) ורק אז מבצע merge (GitHub).
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, json, urllib.request

from policy.rbac import require_perm

router = APIRouter(prefix="/merge_guard", tags=["merge-guard"])

def _post(api: str, path: str, body: dict) -> dict:
    req=urllib.request.Request(api+path, method="POST", data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())

GATE_API = os.environ.get("IMU_GATE_API", "http://127.0.0.1:8000")

class MergeGuardReq(BaseModel):
    user_id: str = "demo-user"
    # Gatekeeper inputs:
    evidences: List[Dict[str,Any]] = Field(default_factory=list)  # [{digest,min_trust}]
    checks: Optional[Dict[str,Any]] = None                        # {owner,repo,ref|pr_number,required,mode,token_env}
    p95: Optional[Dict[str,Any]] = None                           # {keys,ceiling_ms}
    # Merge details:
    owner: str
    repo: str
    pr_number: int
    method: str = "merge"  # merge|squash|rebase
    token_env: str = "GITHUB_TOKEN"

@router.post("/github")
def merge_guard_github(req: MergeGuardReq):
    # RBAC
    require_perm(req.user_id, "gitops:merge_guard")
    # gates:
    gates = _post(GATE_API, "/gatekeeper/evaluate", {
        "evidences": req.evidences,
        "checks": req.checks,
        "p95": req.p95
    })
    if not gates.get("ok"):
        raise HTTPException(412, f"merge_guard_failed: {gates.get('reasons') or 'unknown'}")

    # merge by calling existing handler in guard API (same process via HTTP to avoid coupling)
    payload = {
        "owner": req.owner,
        "repo": req.repo,
        "pr_number": req.pr_number,
        "method": req.method,
        "token_env": req.token_env
    }
    try:
        res = _post(GATE_API, "/gitops/guard/github/merge_pr", payload)
        if not res.get("ok"):
            raise HTTPException(400, f"merge failed: {res}")
        return {"ok": True, "merge": res}
    except Exception as e:
        raise HTTPException(400, f"merge request failed: {e}")