from __future__ import annotations
import os, subprocess, json, urllib.request, tempfile

def _sh(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)

def run(repo_dir: str, title: str, body: str, base: str="main", head_prefix: str="fix/auto") -> dict:
    if os.getenv("AUTO_PR","0").lower() not in ("1","true","yes","on"):
        return {"ok": False, "reason": "AUTO_PR disabled"}
    token = os.getenv("GH_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")  # owner/name
    if not (token and repo):
        return {"ok": False, "reason": "missing GH_TOKEN or GITHUB_REPOSITORY"}

    head = f"{head_prefix}-{os.getpid()}"
    _sh("git","checkout","-b",head, cwd=repo_dir)
    _sh("git","add","-A", cwd=repo_dir)
    # אסוף diff יפה
    diff = _sh("git","diff","--staged","--patch","--unified=3", cwd=repo_dir).stdout
    _sh("git","commit","-m",title, cwd=repo_dir)
    _sh("git","push","-u","origin",head, cwd=repo_dir)

    pr_body = f"{body}\n\n<details><summary>Diff</summary>\n\n```diff\n{diff}\n```\n</details>\n"
    url = f"https://api.github.com/repos/{repo}/pulls"
    data = json.dumps({"title": title, "body": pr_body, "base": base, "head": head}).encode()
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"token {token}", "Content-Type":"application/json"})
    with urllib.request.urlopen(req) as r:
        return {"ok": True, "pr": json.loads(r.read().decode())}
