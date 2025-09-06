# server/main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Body
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any, Optional, List
import asyncio
import json
from pathlib import Path

from server.policy.enforcement import CapabilityPolicy, CapabilityRequest, PolicyError
from server.capabilities.registry import capability_registry
from server.events.bus import EventBus, Topic
from server.pipeline.run_adapter import run_adapter, DryRunError
from server.security.audit import audit_log
from server.security.provenance import ProvenanceStore
from server.state.ttl import TTLRules

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.policy import (
    request_and_continue,
    CapabilityRequest,
    CapabilityResult,
    UserSpacePolicy,
    evaluate_policy_for_user,
)
from engine.provenance import ProvenanceStore, Evidence
from engine.capability_registry import REGISTRY
from adapters import android, ios, unity, cuda, k8s
from server.ws import router as ws_router

app = FastAPI(title="IMU Runtime")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(ws_router, prefix="/ws", tags=["ws"])

prov = ProvenanceStore()

class AdapterPlan(BaseModel):
    adapter: str = Field(..., description="android|ios|unity|cuda|k8s")
    args: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    user_id: str = "default"

class AdapterRunResult(BaseModel):
    ok: bool
    cmd: List[str]
    stdout: str
    stderr: str
    evidence: List[Evidence]

@app.get("/capabilities/list")
def list_caps() -> Dict[str, Any]:
    return {"capabilities": list(REGISTRY.keys())}

@app.post("/capabilities/request", response_model=CapabilityResult)
def capabilities_request(req: CapabilityRequest):
    # Request & Continue: ננסה לבצע התקנה/בדיקה בפועל; אם צריך הרשאות/SDK – נחזיר מצב REQUIRED
    result = request_and_continue(req)
    if result.status == "REQUIRED":
        # רישום ראיות
        prov.add_evidence(
            Evidence(
                claim=f"capability:{req.capability} required",
                source="engine.policy",
                trust=0.9,
                ttl_seconds=3600,
                extra={"missing": result.missing, "hint": result.hint}
            )
        )
    return result

def _get_adapter_impl(name: str):
    name = name.lower()
    if name == "android": return android.AndroidAdapter()
    if name == "ios": return ios.IOSAdapter()
    if name == "unity": return unity.UnityAdapter()
    if name == "cuda": return cuda.CUDAAdapter()
    if name == "k8s": return k8s.K8sAdapter()
    raise HTTPException(400, f"unknown adapter: {name}")

@app.post("/run_adapter", response_model=AdapterRunResult)
def run_adapter(plan: AdapterPlan):
    # מדיניות פר־משתמש (TTL/אימון/ספי אמון) + Provenance קשיח
    policy: UserSpacePolicy = evaluate_policy_for_user(plan.user_id)

    impl = _get_adapter_impl(plan.adapter)
    # DRY RUN מרכיב פקודות ומחזיר בדיוק מה יורץ (עם Evidence)
    cmd = impl.build_command(plan.args, dry_run=plan.dry_run, policy=policy)

    # בדיקת מדיניות קשיחה
    for rule in policy.hard_rules:
        rule.enforce(adapter=plan.adapter, cmd=cmd, args=plan.args)

    stdout, stderr = "", ""
    ok = True
    evidences: List[Evidence] = impl.produce_evidence(cmd, plan.args)

    if not plan.dry_run:
        ok, stdout, stderr = impl.execute(cmd, policy=policy)
        evidences += [
            Evidence(
                claim=f"adapter:{plan.adapter}:exit_status",
                source="adapters."+plan.adapter,
                trust=0.85 if ok else 0.2,
                ttl_seconds=policy.ttl_seconds,
                extra={"ok": ok}
            )
        ]

    # שידור התקדמות ל־WebSocket timeline (גם ב־dry_run)
    from server.ws import push_progress
    push_progress(
        topic=f"run/{plan.user_id}",
        event={"type":"adapter_run", "adapter":plan.adapter, "cmd":cmd, "ok": ok}
    )

    # שמירת ראיות עם תוכן כתובת־תוכן (CAS) וחתימה
    for ev in evidences:
        prov.add_evidence(ev)

    return AdapterRunResult(
        ok=ok, cmd=cmd, stdout=stdout, stderr=stderr, evidence=evidences
    )


# Singletons (in real deployment: DI container)
EVENT_BUS = EventBus()
POLICY = CapabilityPolicy()
PROV = ProvenanceStore(base_dir=Path("./var/provenance"))
TTL = TTLRules()

@app.get("/")
def hello():
    return {"ok": True, "name": "IMU Orchestrator", "version": "1.0.0"}

# ---- WebSocket: progress + timeline stream -------------------------------------------------
active_ws = set()

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    active_ws.add(ws)
    try:
        # Send hello
        await ws.send_text(json.dumps({"type":"hello","msg":"connected"}))
        # Forward events from bus to this ws
        q = EVENT_BUS.subscribe(Topic.TELEMETRY)
        while True:
            event = await q.get()
            try:
                await ws.send_text(json.dumps(event))
            except RuntimeError:
                break
    except WebSocketDisconnect:
        pass
    finally:
        active_ws.discard(ws)

async def _emit(event: Dict[str, Any]):
    # Fan-out to all connected sockets; non-blocking
    dead = []
    for ws in list(active_ws):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_ws.discard(ws)

EVENT_BUS.set_push_hook(_emit)

# ---- Capability “request and continue” API -------------------------------------------------
@app.post("/capabilities/request")
async def request_capability(req: CapabilityRequest, bg: BackgroundTasks):
    """
    מדיניות: מנסים להתקין/להפעיל את היכולת בעת הצורך; אם חסר — מבקשים, ממשיכים ברקע,
    והקריאה מחזירה מייד סטטוס "requested" או "already_available".
    """
    try:
        decision = POLICY.decide(req)
    except PolicyError as e:
        audit_log("capability_request_denied", {"capability": req.name, "reason": str(e)})
        raise HTTPException(status_code=403, detail=f"Denied by policy: {e}")

    if decision == "already_available":
        audit_log("capability_request_ok", {"capability": req.name, "status": "already_available"})
        return {"ok": True, "status": "already_available"}

    # decision == "install"
    cap = capability_registry.resolve(req.name)
    if not cap:
        audit_log("capability_request_missing", {"capability": req.name})
        raise HTTPException(status_code=404, detail=f"Capability '{req.name}' not registered")

    # install in background; client can proceed
    def _install():
        EVENT_BUS.emit(Topic.TELEMETRY, {"type": "capability_install_start", "capability": req.name})
        ok, meta = cap.install()  # may use package managers / download / compile
        PROV.record_capability(req.name, ok, meta)
        EVENT_BUS.emit(Topic.TELEMETRY, {
            "type": "capability_install_done", "capability": req.name, "ok": ok, "meta": meta
        })
        audit_log("capability_install", {"capability": req.name, "ok": ok, "meta": meta})
    bg.add_task(_install)

    return {"ok": True, "status": "requested"}

# ---- Adapter Runner: Dry-run + Run ---------------------------------------------------------
@app.post("/run_adapter/dry")
async def run_adapter_dry(payload: Dict[str, Any]):
    """
    Dry-run: בונה פקודות, בודק Policy/Contracts, ולא מריץ בפועל.
    מחזיר command(s)+env+mounts לצורך שקיפות ובדיקה.
    """
    try:
        plan = await run_adapter(payload, dry=True, event_bus=EVENT_BUS, prov=PROV, ttl=TTL, policy=POLICY)
        return {"ok": True, "plan": plan}
    except DryRunError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run_adapter")
async def run_adapter_exec(payload: Dict[str, Any]):
    """
    Run: מריץ בפועל לאחר בדיקות מדיניות ו־contracts, עם פליטת Telemetry ו־Provenance חתום.
    """
    plan = await run_adapter(payload, dry=False, event_bus=EVENT_BUS, prov=PROV, ttl=TTL, policy=POLICY)
    return {"ok": True, "result": plan}

# ---- Minimal HTML for testing WS (dev DX) --------------------------------------------------
@app.get("/dev/ws")
def dev_ws_page():
    html = """
<!doctype html>
<html><head><meta charset="utf-8"><title>IMU WS</title></head>
<body>
  <h1>IMU Telemetry</h1>
  <pre id="log"></pre>
<script>
const el = document.getElementById('log');
const ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
ws.onmessage = (ev)=>{ el.textContent += ev.data + "\\n"; };
</script>
</body></html>
"""
    return HTMLResponse(html)
