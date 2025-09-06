# server/gitops_api.py
# GitOps API: init/status/remote/branch/commit/push + PR open/list/merge (metadata under .imu/prs)
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from pathlib import Path
import json, time, os, hashlib

from gitops.utils import ensure_repo, status as git_status, set_remote, create_branch, add_all, commit as git_commit, push as git_push, current_branch, merge as git_merge, have_git, install_hint_git
from policy.rbac import require_perm
from policy.policy_hotload import _apply_cfg  # לשימוש כבדיקה (לא משנה state אם לא תחבר)
from provenance.keyring import Keyring
from provenance.envelope import sign_cas_record

router = APIRouter(prefix="/gitops", tags=["gitops"])
PR_DIR = Path(".imu/prs")
PR_DIR.mkdir(parents=True, exist_ok=True)
KR = Keyring(".imu/keys")

class InitReq(BaseModel):
    user_id: str = "demo-user"
    path: str

@router.post("/init")
def init_repo(req: InitReq):
    require_perm(req.user_id, "gitops:init")
    p = ensure_repo(req.path)
    # commit baseline files if any present
    add_all(p, [".gitignore","security/policy_rules.yaml",".imu/keys/pub"])
    try:
        out = git_commit(p, "init repo", author=req.user_id)
    except Exception as e:
        out = f"init commit skipped: {e}"
    return {"ok": True, "path": p, "branch": current_branch(p), "out": out}

class StatusReq(BaseModel):
    path: str

@router.post("/status")
def status_api(req: StatusReq):
    return {"ok": True, "status": git_status(req.path), "branch": current_branch(req.path)}

class RemoteReq(BaseModel):
    user_id: str = "demo-user"
    path: str
    name: str = "origin"
    url: str

@router.post("/remote")
def set_remote_api(req: RemoteReq):
    require_perm(req.user_id, "gitops:remote")
    try:
        out = set_remote(req.path, req.name, req.url)
    except Exception as e:
        raise HTTPException(400, f"set_remote failed: {e}")
    return {"ok": True, "out": out}

class BranchReq(BaseModel):
    user_id: str = "demo-user"
    path: str
    name: str
    base: str = "main"

@router.post("/branch")
def branch_api(req: BranchReq):
    require_perm(req.user_id, "gitops:branch")
    try:
        b = create_branch(req.path, req.name, req.base)
    except Exception as e:
        raise HTTPException(400, f"branch failed: {e}")
    return {"ok": True, "branch": b}

class CommitReq(BaseModel):
    user_id: str = "demo-user"
    path: str
    message: str = "changes"
    add_patterns: List[str] = Field(default_factory=lambda: ["."])

@router.post("/commit")
def commit_api(req: CommitReq):
    require_perm(req.user_id, "gitops:commit")
    add_all(req.path, req.add_patterns)
    out = git_commit(req.path, req.message, author=req.user_id)
    return {"ok": True, "out": out}

class PushReq(BaseModel):
    user_id: str = "demo-user"
    path: str
    remote: str = "origin"
    branch: Optional[str] = None

@router.post("/push")
def push_api(req: PushReq):
    require_perm(req.user_id, "gitops:push")
    if not have_git():
        return {"ok": False, "resource_required": "git", "install": install_hint_git()}
    br = req.branch or current_branch(req.path)
    out = git_push(req.path, req.remote, br)
    return {"ok": True, "out": out}

# ---- Pull Requests (metadata only) ----

class PROpen(BaseModel):
    user_id: str = "demo-user"
    path: str
    branch: str
    target: str = "main"
    title: str = "Update"
    description: str = ""

def _pr_path(pr_id: str) -> Path:
    return PR_DIR / f"{pr_id}.json"

@router.post("/pr/open")
def pr_open(req: PROpen):
    require_perm(req.user_id, "gitops:pr:open")
    # Validate policy file syntax if exists
    pol = Path(req.path)/"security/policy_rules.yaml"
    if pol.exists():
        try:
            import yaml
            with open(pol,"r",encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # יישום זמני של ולידציה (לא משנים state חיצוני אם לא רוצים)
            _apply_cfg(cfg)  # אם תרצה להימנע מכתיבה, אפשר להריץ רק פרסר
        except Exception as e:
            raise HTTPException(400, f"policy validation failed: {e}")

    pr_id = f"pr-{int(time.time())}"
    record = {
        "id": pr_id,
        "branch": req.branch,
        "target": req.target,
        "title": req.title,
        "description": req.description,
        "author": req.user_id,
        "ts": time.time(),
        "repo": os.path.abspath(req.path)
    }
    # חתימת Envelope על רשומת ה-PR (לשם עקיבות)
    priv = KR.load_private()
    kid  = KR.current_kid()
    env  = sign_cas_record(priv, kid, {"type":"pr", **record})
    record["envelope"] = {
        "payloadType": env.payloadType,
        "payload_b64": env.payload_b64,
        "signatures": [s.__dict__ for s in env.signatures]
    }
    _pr_path(pr_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "pr_id": pr_id, "record": record}

@router.get("/pr/list")
def pr_list():
    out=[]
    for p in PR_DIR.glob("pr-*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
    return {"ok": True, "items": sorted(out, key=lambda x:x["ts"], reverse=True)}

class PRMerge(BaseModel):
    user_id: str = "demo-user"
    pr_id: str

@router.post("/pr/merge")
def pr_merge(req: PRMerge):
    require_perm(req.user_id, "gitops:pr:merge")
    p = _pr_path(req.pr_id)
    if not p.exists():
        raise HTTPException(404, "pr not found")
    pr = json.loads(p.read_text(encoding="utf-8"))
    repo = pr["repo"]; branch = pr["branch"]; target = pr["target"]
    # merge locally
    try:
        out = git_merge(repo, target, branch, no_ff=True)
    except Exception as e:
        raise HTTPException(400, f"merge failed: {e}")
    pr["merged_ts"] = time.time()
    p.write_text(json.dumps(pr, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "out": out}