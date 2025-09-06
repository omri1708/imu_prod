# capabilities/manager.py
import subprocess, shlex, json, time
from typing import Dict, Any, Optional
from engine.policy import AskAndProceedPolicy, RequestContext
from engine.provenance import ProvenanceStore, Evidence

class CapabilityError(Exception): ...

class CapabilityManager:
    def __init__(self, policy: AskAndProceedPolicy, prov: ProvenanceStore):
        self.policy = policy
        self.prov = prov
        self.status: Dict[str, Dict[str, Any]] = {}  # name -> {installed:bool, last_attempt:..., msg:...}

    def is_installed(self, name: str) -> bool:
        st = self.status.get(name, {})
        return bool(st.get("installed"))

    def _record(self, name: str, ok: bool, msg: str, commands: list[str], ctx: RequestContext):
        self.status[name] = {
            "installed": ok,
            "last_attempt": time.time(),
            "msg": msg,
            "requested_by": ctx.user.user_id,
        }
        plan = json.dumps({"capability": name, "commands": commands, "ok": ok, "msg": msg}, ensure_ascii=False).encode()
        self.prov.put(Evidence(kind="command_plan", content=plan, meta={"capability": name, "user": ctx.user.user_id}))

    def request(self, name: str, ctx: RequestContext, dry_run: bool = True) -> Dict[str, Any]:
        ent = self.policy.registry.get(name)
        if not ent:
            raise CapabilityError(f"unknown:{name}")
        commands: list[str] = ent.get("installer", [])
        if not commands:
            # ייתכן שזו יכולת פנימית ללא התקנה
            self._record(name, True, "no_install_required", [], ctx)
            return {"ok": True, "installed": True, "msg": "no_install_required"}

        if not self.policy.authorize_install(ctx, name):
            self._record(name, False, "policy_denied", commands, ctx)
            return {"ok": False, "installed": False, "msg": "policy_denied"}

        # הרצה בפועל רק אם dry_run=False
        if dry_run:
            self._record(name, True, "dry_run_ok", commands, ctx)
            return {"ok": True, "installed": False, "msg": "dry_run_ok", "would_run": commands}

        # run with provenance capture
        logs: list[str] = []
        for cmd in commands:
            try:
                proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
                logs.append(f"$ {cmd}\n{proc.stdout}\n{proc.stderr}")
            except subprocess.CalledProcessError as e:
                logs.append(f"$ {cmd}\nEXIT {e.returncode}\n{e.stdout}\n{e.stderr}")
                blob = "\n\n".join(logs).encode()
                self.prov.put(Evidence(kind="installer_log", content=blob, meta={"capability": name, "ok": False}))
                self._record(name, False, f"install_failed:{cmd}", commands, ctx)
                return {"ok": False, "installed": False, "msg": f"install_failed:{cmd}"}

        blob = "\n\n".join(logs).encode()
        self.prov.put(Evidence(kind="installer_log", content=blob, meta={"capability": name, "ok": True}))
        self._record(name, True, "installed", commands, ctx)
        return {"ok": True, "installed": True, "msg": "installed"}