# server/gitops_guard_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os, json, urllib.request, base64, subprocess, shutil

from policy.rbac import require_perm

router = APIRouter(prefix="/gitops/guard", tags=["gitops-guard"])

def _have_git() -> bool: return shutil.which("git") is not None

class VerifyCommitsReq(BaseModel):
    repo_path: str
    rev_range: str = "HEAD~20..HEAD"  # טווח קומיטים לבדיקה

@router.post("/git/verify_signatures")
def verify_commit_signatures(req: VerifyCommitsReq):
    require_perm("demo-user", "gitops:verify")
    if not _have_git(): return {"ok": False, "resource_required": "git"}
    try:
        cmd=["git","log","--show-signature","--pretty=format:%H",req.rev_range]
        p=subprocess.run(cmd, cwd=req.repo_path, text=True, capture_output=True, check=False)
        if p.returncode!=0: raise RuntimeError(p.stdout)
        # בדיקה נאיבית: מחפשים "Good signature" בטקסט
        good = sum(1 for line in p.stdout.splitlines() if "Good signature" in line or "gpg: Good signature" in line)
        total= sum(1 for line in p.stdout.splitlines() if len(line)==40)  # hash lines
        return {"ok": good>=1, "good": good, "total": total}
    except Exception as e:
        raise HTTPException(400, f"verify failed: {e}")

# ---- GitHub PR Guard Rails ----

class GHPRCheckReq(BaseModel):
    owner: str
    repo: str
    pr_number: int
    require_labels: List[str] = Field(default_factory=lambda: ["approved","ready"])
    require_checks_success: bool = True
    token_env: str = "GITHUB_TOKEN"   # איפה מחזיקים טוקן

def _gh_get(path: str, token: str) -> dict:
    req=urllib.request.Request(f"https://api.github.com{path}", headers={"Authorization": f"Bearer {token}", "User-Agent":"imu-gitops"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

@router.post("/github/check_pr")
def github_check_pr(req: GHPRCheckReq):
    require_perm("demo-user", "gitops:pr:guard")
    token=os.environ.get(req.token_env)
    if not token:
        return {"ok": False, "resource_required": req.token_env, "hint":"set a GitHub PAT with repo scope"}
    # labels
    pr=_gh_get(f"/repos/{req.owner}/{req.repo}/issues/{req.pr_number}", token)
    labels=[l["name"] for l in pr.get("labels",[])]
    missing=[x for x in req.require_labels if x not in labels]
    # checks
    checks_ok=True
    if req.require_checks_success:
        # ריכוז סטטוסים — קונקרטי אך פשוט
        statuses=_gh_get(f"/repos/{req.owner}/{req.repo}/commits/{pr['pull_request']['head']['sha']}/status", token)
        checks_ok = statuses.get("state","")=="success"
    return {"ok": not missing and checks_ok, "labels": labels, "missing": missing, "checks_ok": checks_ok}

class GHMergeReq(BaseModel):
    owner: str
    repo: str
    pr_number: int
    method: str = "merge"  # merge|squash|rebase
    token_env: str = "GITHUB_TOKEN"

@router.post("/github/merge_pr")
def github_merge_pr(req: GHMergeReq):
    require_perm("demo-user", "gitops:pr:merge")
    token=os.environ.get(req.token_env)
    if not token:
        return {"ok": False, "resource_required": req.token_env, "hint":"set a GitHub PAT with repo scope"}
    data=json.dumps({"merge_method": req.method}).encode()
    http=urllib.request.Request(f"https://api.github.com/repos/{req.owner}/{req.repo}/pulls/{req.pr_number}/merge",
                               method="PUT", data=data,
                               headers={"Authorization": f"Bearer {token}","Content-Type":"application/json","User-Agent":"imu-gitops"})
    try:
        with urllib.request.urlopen(http, timeout=20) as r:
            return {"ok": True, "status": r.status}
    except Exception as e:
        raise HTTPException(400, f"merge failed: {e}")

# ---- GitLab (מלל דומה; אופציונלי) ----
class GLPRCheckReq(BaseModel):
    base_url: str = "https://gitlab.com/api/v4"
    project_id: str
    mr_iid: int
    token_env: str = "GITLAB_TOKEN"

@router.post("/gitlab/check_mr")
def gitlab_check_mr(req: GLPRCheckReq):
    require_perm("demo-user", "gitops:pr:guard")
    token=os.environ.get(req.token_env)
    if not token:
        return {"ok": False, "resource_required": req.token_env, "hint":"set a GitLab PAT"}
    url=f"{req.base_url}/projects/{req.project_id}/merge_requests/{req.mr_iid}"
    http=urllib.request.Request(url, headers={"PRIVATE-TOKEN": token, "User-Agent":"imu-gitops"})
    with urllib.request.urlopen(http, timeout=20) as r:
        mr=json.loads(r.read().decode())
    return {"ok": mr.get("state") in ("opened","locked"), "mr": mr}