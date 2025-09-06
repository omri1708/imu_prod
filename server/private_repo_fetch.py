# server/private_repo_fetch.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json, urllib.request, urllib.parse, os

def fetch_github_file(owner: str, repo: str, path: str, ref: str, token_env: str = "GITHUB_TOKEN") -> Dict[str,Any]:
    """
    מחזיר {"ok":bool, "content":str, "sha":str} — content הוא הטקסט של הקובץ.
    דורש PAT ב-env אם הפרויקט פרטי.
    """
    token = os.environ.get(token_env)
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}?ref={urllib.parse.quote(ref)}"
    headers = {"Accept":"application/vnd.github+json","User-Agent":"imu-gitops"}
    if token: headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode())
            # אם "content" בקידוד base64—נטפל בזה; לעת עתה GitHub מחזיר base64
            import base64
            text = base64.b64decode(j.get("content","")).decode("utf-8")
            return {"ok": True, "content": text, "sha": j.get("sha")}
    except Exception as e:
        return {"ok": False, "error": str(e), "resource_required": None if token else token_env}

def fetch_gitlab_file(project_id: str, path: str, ref: str, token_env: str = "GITLAB_TOKEN", base_url: str = "https://gitlab.com") -> Dict[str,Any]:
    """
    GitLab raw file: GET /projects/:id/repository/files/:file_path/raw?ref=:branch
    """
    token = os.environ.get(token_env)
    file_path = urllib.parse.quote(path, safe="")
    url = f"{base_url}/api/v4/projects/{urllib.parse.quote(project_id, safe='')}/repository/files/{file_path}/raw?ref={urllib.parse.quote(ref)}"
    headers = {"User-Agent":"imu-gitops"}
    if token: headers["PRIVATE-TOKEN"] = token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            text = r.read().decode("utf-8")
            return {"ok": True, "content": text}
    except Exception as e:
        return {"ok": False, "error": str(e), "resource_required": None if token else token_env}