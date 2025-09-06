# server/http_api.py
from __future__ import annotations
import os, json, hashlib, asyncio, time, subprocess, shlex, platform, shutil
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from security.network_policies import is_allowed, POLICY_DB
from security.filesystem_policies import is_path_allowed, cleanup_ttl, FS_DB
from adapters.mappings import WINGET, BREW, APT, CLI_TEMPLATES
from runtime.p95 import GATES
from server.stream_wfq import BROKER  # WFQ Broker
from policy.rbac import require_perm

APP = FastAPI(title="IMU Adapter API")

from policy.policy_hotload import start_watcher
start_watcher("security/policy_rules.yaml", interval_s=2.0)

from server.provenance_api import router as prov_router
from server.metrics_api import router as metrics_router
from server.supplychain_api import router as supply_router
from server.stream_gateway import router as events_router
from server.supplychain_index_api import router as sc_index_router
from server.runbook_api import router as runbook_router
from server.key_admin_api import router as key_admin_router
from server.archive_api import router as archive_router
from server.bundles_api import router as bundles_router
from server.unified_archive_api import router as unified_router
from server.replay_api import router as replay_router
from server.gitops_api import router as gitops_router
from server.audit_ops import AuditMiddleware
from server.gitops_guard_api import router as guard_router
from server.gitops_checks_api import router as gh_checks_router
from server.policy_edit_api import router as policy_edit_router
from server.webhooks_api import router as webhooks_router
from server.gatekeeper_api import router as gatekeeper_router

APP.include_router(gatekeeper_router)
APP.include_router(webhooks_router)
APP.include_router(policy_edit_router)
APP.include_router(gh_checks_router)
APP.include_router(guard_router)
APP.add_middleware(AuditMiddleware)
APP.include_router(gitops_router)
APP.include_router(replay_router)
APP.include_router(unified_router)
APP.include_router(bundles_router)
APP.include_router(archive_router)
APP.include_router(key_admin_router)
APP.include_router(prov_router)
APP.include_router(metrics_router)
APP.include_router(supply_router)
APP.include_router(events_router)
APP.include_router(sc_index_router)
APP.include_router(runbook_router)


# ---------- Utils ----------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _os_family() -> str:
    sysname = platform.system().lower()
    if "windows" in sysname: return "win"
    if "darwin" in sysname: return "mac"
    return "linux"

def _p95_ceiling_ms(user_id: str, route: str) -> int:
    # אפשר להרחיב לפי משתמש ממש; כאן ceiling ברירת מחדל
    return int(os.environ.get("IMU_P95_MS", "5000"))

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

# ---------- Network Policy dump ----------
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

# ---------- Capability request ----------

class CapabilityRequest(BaseModel):
    user_id: str
    capability: str   # e.g., "unity.hub", "jdk", "gradle", "kubectl", "cuda"
    
@APP.post("/capabilities/request")
async def request_capability(req: CapabilityRequest):
    require_perm(req.user_id, "capabilities:request")
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

    if fam == "win":
        cmd = f"winget install -e --id {mp}"
    elif fam == "mac":
        cmd = f"brew install {mp}"
    else:
        cmd = f"sudo apt-get update && sudo apt-get install -y {mp}"

    ev = Evidence(kind="install_command",
                  content_sha256=sha256_bytes(cmd.encode()),
                  source=f"mapping:{fam}",
                  trust=0.7)
    BROKER.ensure_topic("timeline", rate=50, burst=200, weight=2)
    BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":f"capability.request {req.capability}"}, priority=5)
    return {"ok": True, "command": cmd, "evidence": [ev.dict()]}

# ---------- Adapters: dry_run / run ----------
class DryRunRequest(BaseModel):
    user_id: str
    kind: str          # "unity.build" | "android.gradle" | "ios.xcode" | "k8s.kubectl.apply" | "cuda.nvcc"
    params: Dict[str, Any]

@APP.post("/adapters/dry_run", response_model=RunResult)
async def adapters_dry_run(req: DryRunRequest):
    require_perm(req.user_id, f"adapter:dry_run:{req.kind}")
    fam = _os_family()
    tmpl_map = CLI_TEMPLATES.get(req.kind)
    if not tmpl_map:
        raise HTTPException(400, "unknown kind")
    tmpl = tmpl_map.get(fam) or tmpl_map.get("any")
    if not tmpl:
        raise HTTPException(400, "unsupported on this OS")

    # Compose deterministically
    try:
        cmd = tmpl.format(**req.params)
    except KeyError as e:
        return RunResult(ok=False, cmd="", reason=f"missing_param:{e.args[0]}", evidence=[])

    # Hard deny tokens
    forbidden_tokens = [" rm -rf ", " :(){", "mkfs", " dd if=", ";rm -rf", "&& rm -rf"]
    if any(t in f" {cmd} " for t in forbidden_tokens):
        return RunResult(ok=False, cmd=cmd, reason="blocked_by_policy", evidence=[])

    # FS gating on known param keys
    path_keys_read  = ["project", "workspace", "manifest", "src", "log"]
    path_keys_write = ["out", "keystore"]
    for k in path_keys_read:
        if k in req.params and not is_path_allowed(req.user_id, str(req.params[k]), write=False):
            return RunResult(ok=False, cmd=cmd, reason=f"fs_denied_read:{k}", evidence=[])
    for k in path_keys_write:
        if k in req.params and not is_path_allowed(req.user_id, str(req.params[k]), write=True):
            return RunResult(ok=False, cmd=cmd, reason=f"fs_denied_write:{k}", evidence=[])

    ev = Evidence(kind="cli-template",
                  content_sha256=sha256_bytes(cmd.encode()),
                  source=f"template:{req.kind}",
                  trust=0.9)
    BROKER.ensure_topic("timeline", rate=50, burst=200, weight=2)
    BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":f"dry_run {req.kind}"}, priority=3)
    return RunResult(ok=True, cmd=cmd, evidence=[ev])

class RunAdapterRequest(BaseModel):
    user_id: str
    kind: str
    params: Dict[str, Any]
    execute: bool = False  # True => להריץ בפועל

@APP.post("/adapters/run", response_model=RunResult)
async def adapters_run(req: RunAdapterRequest):
    require_perm(req.user_id, f"adapter:dry_run:{req.kind}")
    t0 = time.time()
    BROKER.ensure_topic("timeline", rate=50, burst=200, weight=2)
    BROKER.ensure_topic("progress", rate=80, burst=400, weight=3)
    BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":f"run {req.kind} start"}, priority=2)

    # First dry-run
    dry = await adapters_dry_run(DryRunRequest(user_id=req.user_id, kind=req.kind, params=req.params))
    if not dry.ok:
        return dry
    if not req.execute:
        BROKER.submit("progress","api",{"type":"progress","ts":time.time(),"pct":50,"note":"dry_run_only"}, priority=6)
        ms = (time.time()-t0)*1000
        GATES.observe(f"adapters.run:{req.kind}", ms)
        try:
            GATES.ensure(f"adapters.run:{req.kind}", _p95_ceiling_ms(req.user_id, "adapters.run"))
        except Exception as e:
            BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":str(e)}, priority=1)
        return dry

    # Execute if binary present
    bin_name = dry.cmd.split()[0]
    if not shutil.which(bin_name):
        cap = req.kind.split('.', 1)[0]
        cmd_req = await request_capability(CapabilityRequest(user_id=req.user_id, capability=cap))
        evs = [Evidence(**e) for e in cmd_req["evidence"]] if isinstance(cmd_req, dict) and "evidence" in cmd_req else []
        BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":f"resource_required {cap}"}, priority=1)
        return RunResult(ok=False, cmd=dry.cmd, reason="resource_required", evidence=evs)

    BROKER.submit("progress","api",{"type":"progress","ts":time.time(),"pct":10,"note":"exec_start"}, priority=6)
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
        BROKER.submit("progress","api",{"type":"progress","ts":time.time(),"pct":95,"note":"exec_done"}, priority=6)
        ms = (time.time()-t0)*1000
        GATES.observe(f"adapters.run:{req.kind}", ms)
        try:
            GATES.ensure(f"adapters.run:{req.kind}", _p95_ceiling_ms(req.user_id, "adapters.run"))
        except Exception as e:
            BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":str(e)}, priority=1)
        BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":f"run {req.kind} finish ok={ok}"}, priority=2)
        return RunResult(ok=ok, cmd=dry.cmd, reason=None if ok else f"exit_{proc.returncode}", evidence=[ev])
    except Exception as e:
        BROKER.submit("timeline","api",{"type":"event","ts":time.time(),"note":f"exec_failed {e}"}, priority=1)
        raise HTTPException(500, f"exec_failed: {e}")