# imu_repo/engine/pipeline.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Tuple, Optional
import time as _t


from grounded.fact_gate import FactGate, EvidenceIndex, SchemaRule, UnitRule, FreshnessRule, FactGatePolicy
from grounded.validators import TrustRule, ApiRule, ConsistencyRule, ValidationError as GroundValidationError

from grounded.provenance_store import ProvenanceStore
from grounded.audit import AuditLog
from core.contracts.verifier import Contracts, ContractViolation
from core.vm.vm import VMError
from core.vm.vm import VM 
from adapters.fs_sandbox import FSSandbox
from adapters.http_fetch import http_fetch
from adapters.async_tasks import AsyncPool
from adapters.net_sandbox import NetSandbox
from adapters.db_localqueue import LocalQueue

from user.memory_state import MemoryState
from user.consciousness import UserConsciousness, UserMind
from user.consolidation import Consolidation
from obs.kpi import KPI
from user.auth import UserStore
from engine.exec_api import exec_best


# TODO
# - חיבור ל־Engine (נקודת כניסה)
# הוסף ל־engine/pipeline.py (בפונקציה שמטפלת בבקשות) מסלול להרצת תא־קוד:

# בתוך engine/pipeline.py (במקום מתאים במסלול HTTP/CLI)
# from engine.exec_api import exec_best
# ...
# if request["type"] == "exec_cell":
#     result = exec_best(request["task"], ctx=request.get("ctx",{}))
#     return 200, {"result": result}


class ResourceRequired(Exception):
    def __init__(self, what:str, how:str): self.what=what; self.how=how


class Engine:
    """Main engine orchestrating program execution with grounding, contracts, and audit."""

    def __init__(self, root: str | None = None):
        root = root or os.getenv("IMU_ROOT", ".imu_state")
        self.store = ProvenanceStore(f"{root}/prov")
        self.idx   = EvidenceIndex(self.store)
        self.mem  = MemoryState(root=f"{root}/memory")
        # TODO שים לב: password כאן דיפולטיבי; החלף לפי סביבה/מפתח
        self.ucon = UserConsciousness(root=f"{root}/consciousness", password="imu", strict_security=False)
        self.cons = Consolidation(self.mem, self.ucon)
        self.kpi = KPI()
        self.user_store = UserStore()

        # חוקים: סכימה, עדכניות, אמינות (trust), אימות API (אם יש), קונסיסטנטיות
        self.gate  = FactGate(self.idx, rules=[
            SchemaRule(),
            FreshnessRule(max_age_seconds=86400),
            TrustRule(min_trust=0.6),
            ApiRule(allow_hosts=None),   # ניתן לספק allowlist ב-run-time
            ConsistencyRule()
        ])
        self.audit = AuditLog(f"{root}/audit.log")
        self.contracts = Contracts()

        # core adapters
        self.fs    = FSSandbox(base=f"{root}/fs", readonly=False)
        self.pool  = AsyncPool(max_workers=8)
        self.net   = NetSandbox()
        self.queue = LocalQueue(f"{root}/queue")

        # register capabilities for VM
        self.caps = {
            "fs_read": self.fs.read_text,
            "fs_write": self.fs.write_text,
            "http_fetch": http_fetch,
            "queue_put": self.queue.put,
            "queue_get": self.queue.get,
            "async_submit": self.pool.submit,
            "async_result": self.pool.result,
            "net_open": self.net.open,
            "net_close": self.net.close,
        }

    def run_program(
        self,
        program: List[Dict[str, Any]],
        payload: Dict[str, Any],
        policy: str = "strict",
        api_allow_hosts: List[str] | None = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:

        ctx = dict(ctx or {})
        user_id = ctx.get("user_id") or payload.get("user_id", "anon")
        self.user_store.ensure_user(user_id)
        mind = UserMind(user_id, MemoryState(user_id))
        ctx["__mind__"] = mind
        ctx["__routing_hints__"] = mind.routing_hints()
        # Gate מקומי לריצה הזו (לא נוגע בכללים של המופע המשותף)
        gate = FactGate(
            self.idx,
            rules=[
                SchemaRule(),
                FreshnessRule(max_age_seconds=86400),
                TrustRule(min_trust=0.6),
                ApiRule(allow_hosts=api_allow_hosts or []),
                ConsistencyRule(),
            ],
            store=self.gate.store,
            registry=self.gate.registry,
        )
        ctx["payload"] = payload
        ctx["__enforce_in_vm__"] = False      
        _t0 = _t.time()
        
        try:
            result, metrics = VM.run(program, ctx, {"caps": self.caps},
                                     cpu_steps_max=500000,
                                     mem_kb_max=65536,
                                     io_calls_max=10000,
                                     max_sleep_ms=1000)
        except VMError as e:
            self.audit.append("vm_error", {"error": str(e)})
            return 500, {"error":"vm_error","reason":str(e)}

        # חוזה משאבים
        try:
            self.contracts.check_resources(metrics,
                                           limits={"cpu_steps_max":500000,
                                                   "mem_kb_max":65536,
                                                   "io_calls_max":10000})
        except ContractViolation as cv:
            self.audit.append("contract_violation", {"type": cv.kind, "detail": cv.detail})
            return 429, {"error":"contract_violation","kind":cv.kind,"detail":cv.detail}

        body   = result.get("body", {})
        claims = result.get("claims", [])
        
        # 🔒 enforcement: ב־strict חובה שתהיינה claims; אחרת — 422
        if policy == "strict" and not claims:
            self.audit.append("no_claims_reject", {"policy":"strict"})
            return 422, {"error":"no_claims","reason":"strict_policy_requires_claims"}

        # אכיפת Grounding + Trust + API
        if claims:
            ok, diags = gate.check_claims(claims, strict=True)
            self.audit.append("claims_checked", {"claims": claims, "ok": ok, "diags": diags})
            if not ok:
                return 422, {"error": "invalid_or_weak_evidence", "diags": diags}
            body["_provenance"] = [d.get("prov_id") for d in diags if isinstance(d,dict) and d.get("prov_id")]

        code, body = int(result.get("status",200)), body

        # 🔒 Engine-level enforcement: חוסם גם אם מנסים לעקוף את האכיפה בתוך ה-VM
        if policy == "strict":
            try:
                # ודא של-ctx יש את ה-claims/‏registers כך שה-gate יוכל לבדוק הקשרים
                ctx.setdefault("__claims__", claims)
                ctx.setdefault("__registers__", {})  # אופציונלי — אם תרצה להעביר רגיסטרים מה-VM
                # אפשר להעביר FactGatePolicy מותאם דרך ctx["__fact_policy__"], אם אין — דיפולט
                gate.enforce(ctx, body, ctx.get("__fact_policy__"))
            except GroundValidationError as e:
                self.audit.append("engine_fact_gate_block", {"detail": str(e)})
                code, body = 412, {"error":"precondition_failed","detail":str(e)}

        # ✨ Post-run consolidation & user-consciousness
        user_id = payload.get("user_id","default")
        text_obs = payload.get("utterance") or payload.get("query") or str(payload)[:512]
        derived_prefs = body.get("_derived_prefs") if isinstance(body, dict) else None
        try:
            self.cons.on_interaction(user_id, text_obs, derived_prefs if isinstance(derived_prefs, dict) else None)
        except Exception as e:
            self.audit.append("consolidation_warn", {"reason": str(e)})

        _lat_ms = (_t.time() - _t0) * 1000.0
        self.kpi.record(latency_ms=_lat_ms, error=not (200 <= code < 400))
                
        return code, body
     

def bootstrap_complete_system() -> Engine:
    return Engine()

if __name__=="__main__":
    eng = bootstrap_complete_system()
    prog = [
        {"op":"PUSH","ref":"$.payload.a"},
        {"op":"PUSH","ref":"$.payload.b"},
        {"op":"ADD"},
        {"op":"STORE","reg":"sum"},
        {"op":"EVIDENCE","claim":"a+b=sum","sources":["unit:test:add"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:sum"}}
    ]
    code, out = eng.run_program(prog, {"a":2,"b":3})
    print(code, json.dumps(out, ensure_ascii=False))
