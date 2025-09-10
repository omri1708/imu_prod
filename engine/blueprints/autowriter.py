from __future__ import annotations
import os, json, subprocess, shlex, tempfile, time
from typing import Dict, Any

REG = "var/registry/blueprints.json"
os.makedirs(os.path.dirname(REG), exist_ok=True)

def _git(cmd: str, cwd: str="."):
    return subprocess.call(shlex.split(cmd), cwd=cwd)

def write_blueprint(*, name: str, code: str, tests: str, registry_meta: Dict[str,Any]) -> Dict[str,Any]:
    mod_path = f"blueprints/{name}.py"; test_path = f"tests/test_{name}.py"
    os.makedirs("blueprints", exist_ok=True)
    open(mod_path,"w",encoding="utf-8").write(code)
    os.makedirs("tests", exist_ok=True)
    open(test_path,"w",encoding="utf-8").write(tests)
    # עדכון רג'יסטרי
    if os.path.exists(REG):
        d=json.loads(open(REG,"r",encoding="utf-8").read())
    else:
        d={"items":[]}
    d["items"].append({"name":name, "file": mod_path, "ts": time.time(), **registry_meta})
    open(REG,"w",encoding="utf-8").write(json.dumps(d,ensure_ascii=False,indent=2))
    return {"ok": True, "module": mod_path, "test": test_path, "registry": REG}

def create_pr(branch: str, title: str, body: str="") -> Dict[str,Any]:
    # DRYRUN אם git לא מוגדר
    _git("git checkout -b "+branch) or _git("git add .") or _git('git commit -m "{}"'.format(title))
    _git("git push -u origin "+branch)
    # פתיחת PR דרך gh CLI אם קיים
    if _git("gh pr create --title '{}' --body '{}'".format(title, body)) == 0:
        return {"ok": True, "pr": "created via gh"}
    return {"ok": True, "pr": "pushed branch only"}
