# server/http_api.py
# FastAPI:
#  - /adapters/dry_run  (הרכבת פקודה דטרמיניסטית + provenance + gating ראשוני)
#  - /adapters/run      (אותה פקודה; ריצה אמיתית אם execute=True, כולל evidence של הפלט)
#  - /capabilities/request  (מדיניות "לבקש ולהמשיך": פקודת התקנה מדויקת לכל OS/מנג'ר)
#  - /api/policy/network/{user_id}  (הצגת פוליסי רשת פעיל)

from __future__ import annotations
import os, json, hashlib, asyncio, time, subprocess, shlex, platform, shutil
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from security.network_policies import is_allowed, POLICY_DB
from security.filesystem_policies import is_path_allowed, cleanup_ttl, FS_DB
from adapters.mappings import WINGET, BREW, APT, CLI_TEMPLATES

APP = FastAPI(title="IMU Adapter API")

# ---------- Models ----------

class Evidence(BaseModel):
    kind: str
    content_sha256: str
    source: str
    trust: float = Field(ge=0.0, le=1.0)

class RunResult(BaseModel):
    ok: bool
    cmd: str
    reason: Optional[str] = None
    evidence: List[Evidence] = []

# ---------- Utils ----------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _os_family() -> str:
    sysname = platform.system().lower()
    if "windows" in sysname: return "win"
    if "darwin" in sysname: return "mac"
    return "linux"

# ---------- Endpoints ----------

@APP.get("/api/policy/network/{user_id}")
async def get_net_policy(user_id: str):
    p = POLICY_DB.get(user_id)
    if not p: raise HTTPException(404, "no policy")
    return JSONResponse(content={
        "default_deny": p.default_deny,
        "rules": [r.__dict__ for r in p.rules],
        "max_outbound_qps": p.max_outbound_qps,
        "max_concurrent": p.max_concurrent,
    })

class CapabilityRequest(BaseModel):
    user_id: str
    capability: str   # e.g., "unity.hub", "jdk", "gradle", "kubectl", "cuda"

@APP.post("/capabilities/request")
async def request_capability(req: CapabilityRequest):
    fam = _os_family()
    if fam == "win":
        mp = WINGET.get(req.capability)
    elif fam == "mac":
        mp = BREW.get(req.capability)
    else:
        mp = APT.get(req.capability)

    if not mp:
        return JSONResponse(status_code=400, content={
            "ok": False, "error": "unknown_capability", "capability": req.capability
        })

    # לא מתקינים אוטומטית כאן (רשאות/מדיניות); מחזירים פקודה מדויקת.
    if fam == "win":
        cmd = f"winget install -e --id {mp}"
    elif fam == "mac":
        # brew cask אם צריך (במיפויים עצמם כבר מוגדר מה מותקן דרך cask)
        cmd = f"brew install {mp}"
    else:
        cmd = f"sudo apt-get update && sudo apt-get install -y {mp}"

    ev = Evidence(kind="install_command",
                  content_sha256=sha256_bytes(cmd.encode()),
                  source=f"mapping:{fam}",
                  trust=0.7)
    return {"ok": True, "command": cmd, "evidence": [ev.dict()]}

class DryRunRequest(BaseModel):
    user_id: str
    kind: str          # "unity.build" | "android.gradle" | "ios.xcode" | "k8s.kubectl.apply" | "cuda.nvcc"
    params: Dict[str, Any]

@APP.post("/adapters/dry_run", response_model=RunResult)
async def adapters_dry_run(req: DryRunRequest):
    fam = _os_family()
    tmpl_map = CLI_TEMPLATES.get(req.kind)
    if not tmpl_map:
        raise HTTPException(400, "unknown kind")
    tmpl = tmpl_map.get(fam) or tmpl_map.get("any")
    if not tmpl:
        raise HTTPException(400, "unsupported on this OS")

    # הרכבת הפקודה דטרמיניסטית
    try:
        cmd = tmpl.format(**req.params)
    except KeyError as e:
        return RunResult(ok=False, cmd="", reason=f"missing_param:{e.args[0]}", evidence=[])

    # חסימת טוקנים מסוכנים באופן קשיח
    forbidden_tokens = [" rm -rf ", " :(){", "mkfs", " dd if=", ";rm -rf", "&& rm -rf"]
    if any(t in f" {cmd} " for t in forbidden_tokens):
        return RunResult(ok=False, cmd=cmd, reason="blocked_by_policy", evidence=[])

    # Gating בסיסי לפי FS policy על פרמטרים ידועים
    path_keys_read  = ["project", "workspace", "manifest", "src", "log"]
    path_keys_write = ["out", "keystore"]
    for k in path_keys_read:
        if k in req.params:
            p = str(req.params[k])
            if not is_path_allowed(req.user_id, p, write=False):
                return RunResult(ok=False, cmd=cmd, reason=f"fs_denied_read:{k}", evidence=[])
    for k in path_keys_write:
        if k in req.params:
            p = str(req.params[k])
            if not is_path_allowed(req.user_id, p, write=True):
                return RunResult(ok=False, cmd=cmd, reason=f"fs_denied_write:{k}", evidence=[])

    ev = Evidence(kind="cli-template",
                  content_sha256=sha256_bytes(cmd.encode()),
                  source=f"template:{req.kind}",
                  trust=0.9)
    return RunResult(ok=True, cmd=cmd, evidence=[ev])

class RunAdapterRequest(BaseModel):
    user_id: str
    kind: str
    params: Dict[str, Any]
    execute: bool = False  # True => להריץ בפועל

@APP.post("/adapters/run", response_model=RunResult)
async def adapters_run(req: RunAdapterRequest):
    # ראשית dry-run לאותה פקודה, להבטיח דטרמיניזם וגייטינג
    dry = await adapters_dry_run(DryRunRequest(user_id=req.user_id, kind=req.kind, params=req.params))
    if not dry.ok:
        return dry

    if not req.execute:
        return dry  # מחזירים את ההרכבה + evidence, בלי להריץ

    # בדיקת binary קיים
    bin_name = dry.cmd.split()[0]
    if not shutil.which(bin_name):
        # מציעים פקודת התקנה מדויקת לפי ה־capability הראשי
        cap = req.kind.split('.', 1)[0]
        cmd_req = await request_capability(CapabilityRequest(user_id=req.user_id, capability=cap))
        evs = [Evidence(**e) for e in cmd_req["evidence"]] if isinstance(cmd_req, dict) and "evidence" in cmd_req else []
        return RunResult(ok=False, cmd=dry.cmd, reason="resource_required", evidence=evs)

    # ריצה אמיתית (ללא מוקים) + איסוף evidence של הפלט
    try:
        proc = await asyncio.create_subprocess_shell(
            dry.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await proc.communicate()
        ok = (proc.returncode == 0)
        ev = Evidence(kind="process_output",
                      content_sha256=sha256_bytes(out or b""),
                      source="local_run",
                      trust=0.8)
        return RunResult(ok=ok, cmd=dry.cmd, reason=None if ok else f"exit_{proc.returncode}", evidence=[ev])
    except Exception as e:
        raise HTTPException(500, f"exec_failed: {e}")