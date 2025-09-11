# server/webhooks_api.py  (UPDATED)
from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException, Request
from typing import Dict, Any, Optional, List
import hmac, hashlib, json, time, os, urllib.parse, base64

from server.stream_wfq import BROKER
from server.pac_pipeline import pac_run, POLICY_FILE
from server.private_repo_fetch import fetch_github_file, fetch_gitlab_file

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

def _broadcast(note: str, kind: str = "event", pct: float | None = None, priority: int = 4):
    BROKER.ensure_topic("timeline", rate=100, burst=500, weight=2)
    ev={"type": kind, "ts": time.time(), "note": note}
    if pct is not None: ev["pct"]=pct
    BROKER.submit("timeline","webhook",ev, priority=priority)

def _git_sha256(secret: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), msg=body, digestmod=hashlib.sha256)
    return "sha256=" + mac.hexdigest()

# ---------- GitHub ----------
@router.post("/github")
async def github(request: Request,
                 x_github_event: str = Header(None),
                 x_hub_signature_256: str = Header(None)):
    body = await request.body()
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET")
    if secret:
        expected = _git_sha256(secret, body)
        if not hmac.compare_digest(expected, x_hub_signature_256 or ""):
            raise HTTPException(401, "bad signature")
    try:
        payload = json.loads(body.decode() or "{}")
    except Exception:
        raise HTTPException(400, "bad json")

    event = x_github_event or payload.get("action","unknown")
    _broadcast(f"github:{event}")

    # push → PaC: ננסה להביא policy מה-API במקום להסתמך על קובץ מקומי
    if event == "push":
        repo = payload.get("repository",{})
        owner = repo.get("owner",{}).get("name") or repo.get("owner",{}).get("login","")
        name  = repo.get("name","")
        ref   = payload.get("after") or payload.get("ref","main").split("/")[-1]
        changed = set()
        for c in payload.get("commits", []):
            for arr in ("added","modified"):
                for f in c.get(arr,[]):
                    changed.add(f)
        if "security/policy_rules.yaml" in changed:
            _broadcast(f"pac.detected policy change @ {owner}/{name}@{ref}", priority=3)
            fetch = fetch_github_file(owner, name, "security/policy_rules.yaml", ref)
            if fetch.get("ok"):
                try:
                    res = pac_run(fetch["content"], user_id="github-hook", sign_bundle=True)
                    _broadcast(f"pac.applied ms={res['ms']}", priority=3)
                except Exception as e:
                    _broadcast(f"pac.failed: {e}", priority=2)
            else:
                # fallback: קובץ מקומי
                try:
                    y = POLICY_FILE.read_text(encoding="utf-8")
                    res = pac_run(y, user_id="github-hook", sign_bundle=True)
                    _broadcast(f"pac.applied (local) ms={res['ms']}", priority=3)
                except Exception as e:
                    _broadcast(f"pac.failed(local): {e}", priority=2)

    if event == "pull_request":
        action = payload.get("action")
        pr_num = payload.get("number")
        _broadcast(f"pr #{pr_num} {action}", priority=4)

    return {"ok": True}

# ---------- GitLab ----------
@router.post("/gitlab")
async def gitlab(request: Request,
                 x_gitlab_token: str = Header(None),
                 x_gitlab_event: str = Header(None)):
    body = await request.body()
    secret = os.environ.get("GITLAB_WEBHOOK_TOKEN")
    if secret and (x_gitlab_token or "") != secret:
        raise HTTPException(401, "bad token")

    try:
        payload = json.loads(body.decode() or "{}")
    except Exception:
        raise HTTPException(400, "bad json")

    event = x_gitlab_event or payload.get("object_kind","unknown")
    _broadcast(f"gitlab:{event}")

    if event == "push":
        project = payload.get("project",{})
        pid = str(project.get("id",""))
        ref = payload.get("after") or (payload.get("ref") or "main").split("/")[-1]
        files=[]
        for c in payload.get("commits", []):
            files += (c.get("added",[])+c.get("modified",[]))
        if "security/policy_rules.yaml" in files:
            _broadcast(f"pac.detected policy change @ {pid}@{ref}", priority=3)
            fetch = fetch_gitlab_file(pid, "security/policy_rules.yaml", ref)
            if fetch.get("ok"):
                try:
                    res = pac_run(fetch["content"], user_id="gitlab-hook", sign_bundle=True)
                    _broadcast(f"pac.applied ms={res['ms']}", priority=3)
                except Exception as e:
                    _broadcast(f"pac.failed: {e}", priority=2)
            else:
                try:
                    y = POLICY_FILE.read_text(encoding="utf-8")
                    res = pac_run(y, user_id="gitlab-hook", sign_bundle=True)
                    _broadcast(f"pac.applied (local) ms={res['ms']}", priority=3)
                except Exception as e:
                    _broadcast(f"pac.failed(local): {e}", priority=2)

    return {"ok": True}