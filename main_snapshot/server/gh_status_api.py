# server/gh_status_api.py
# GitHub Status Context: יצירת סטטוס על commit לפי החלטת Gatekeeper (success/failure/pending)
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os, json, urllib.request

from policy.rbac import require_perm

router = APIRouter(prefix="/status/github", tags=["github-status"])

class StatusReq(BaseModel):
    user_id: str = "demo-user"
    owner: str
    repo: str
    sha: str
    state: str = Field(..., regex="^(success|failure|error|pending)$")
    context: str = Field("IMU/Gatekeeper")
    description: str = Field("", max_length=140)
    target_url: Optional[str] = None
    token_env: str = "GITHUB_TOKEN"

def _gh_post_status(token: str, owner: str, repo: str, sha: str, body: Dict[str,Any]) -> Dict[str,Any]:
    url=f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"
    req=urllib.request.Request(url, method="POST", data=json.dumps(body).encode(),
                               headers={"Authorization": f"Bearer {token}",
                                        "Accept":"application/vnd.github+json",
                                        "User-Agent":"imu-gitops"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

@router.post("/set")
def set_status(req: StatusReq):
    require_perm(req.user_id, "gitops:status:set")
    token=os.environ.get(req.token_env)
    if not token:
        return {"ok": False, "resource_required": req.token_env, "hint": "set a GitHub PAT with repo:status scope"}
    body={"state":req.state,"context":req.context,"description":req.description}
    if req.target_url: body["target_url"]=req.target_url
    try:
        out=_gh_post_status(token, req.owner, req.repo, req.sha, body)
        return {"ok": True, "response": out}
    except Exception as e:
        raise HTTPException(400, f"set status failed: {e}")