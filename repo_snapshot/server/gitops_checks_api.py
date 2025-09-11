# server/gitops_checks_api.py
# Guard rails מפורטים מול GitHub Checks API v3:
# - /gitops/guard/github/checks  : מביא רשימת check-runs והסטטוס שלהן עבור ref
# - /gitops/guard/github/require : בודק רשימת checks נדרשת לפי PR number או ref
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os, json, urllib.request

from policy.rbac import require_perm

router = APIRouter(prefix="/gitops/guard/github", tags=["gitops-guard-gh"])

def _gh(token: str, path: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "imu-gitops"
        })
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())

class ChecksReq(BaseModel):
    owner: str
    repo: str
    ref: str
    token_env: str = "GITHUB_TOKEN"

@router.post("/checks")
def checks(req: ChecksReq):
    require_perm("demo-user", "gitops:pr:guard")
    token = os.environ.get(req.token_env)
    if not token:
        return {"ok": False, "resource_required": req.token_env, "hint": "export a GitHub token (repo scope)"}
    # GitHub Checks API
    runs = _gh(token, f"/repos/{req.owner}/{req.repo}/commits/{req.ref}/check-runs")
    # Commit Statuses (aggregated)
    status = _gh(token, f"/repos/{req.owner}/{req.repo}/commits/{req.ref}/status")
    checks = [{"name": cr["name"],
               "status": cr["status"],
               "conclusion": cr["conclusion"],
               "started_at": cr.get("started_at"),
               "completed_at": cr.get("completed_at")}
              for cr in runs.get("check_runs",[])]
    aggregate = status.get("state","unknown")
    return {"ok": True, "aggregate": aggregate, "checks": checks}

class RequireReq(BaseModel):
    owner: str
    repo: str
    pr_number: Optional[int] = None
    ref: Optional[str] = None
    require: List[str] = Field(default_factory=list)
    mode: str = Field("all", description="all|any")
    token_env: str = "GITHUB_TOKEN"

@router.post("/require")
def require_checks(req: RequireReq):
    require_perm("demo-user", "gitops:pr:guard")
    token = os.environ.get(req.token_env)
    if not token:
        return {"ok": False, "resource_required": req.token_env, "hint": "export a GitHub token (repo scope)"}
    if not (req.pr_number or req.ref):
        raise HTTPException(400, "must provide pr_number or ref")
    if req.pr_number:
        pr = _gh(token, f"/repos/{req.owner}/{req.repo}/pulls/{req.pr_number}")
        ref = pr["head"]["sha"]
    else:
        ref = req.ref
    runs = _gh(token, f"/repos/{req.owner}/{req.repo}/commits/{ref}/check-runs").get("check_runs",[])
    summary = {cr["name"]: (cr["conclusion"] or cr["status"]) for cr in runs}
    missing = [n for n in req.require if n not in summary]
    bad = [n for n,v in summary.items() if (n in req.require or not req.require) and v not in ("success","neutral","skipped")]
    ok = (len(missing)==0) and \
         ((req.mode=="all" and len(bad)==0) or (req.mode=="any" and len(bad)<len(summary)))
    return {"ok": ok, "ref": ref, "missing": missing, "failing": bad, "summary": summary}