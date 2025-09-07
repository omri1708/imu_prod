# -*- coding: utf-8 -*-
from __future__ import annotations
import json, shlex, asyncio, re
from typing import Dict, Any, List, Optional
from pathlib import Path

from assurance.assurance import AssuranceKernel
from assurance.signing import Signer
from assurance.validators import schema_validator
from assurance.errors import ResourceRequired, RefusedNotGrounded, ValidationFailed
from executor.sandbox import SandboxExecutor, Limits
from executor.policy import Policy
from user_model.model import UserStore, UserModel

# --- Resolver לקליטת תבנית CLI עבור kind ---
def _resolve_template(kind: str) -> Optional[str]:
    # 1) dynamic registry (synth adapters)
    try:
        from adapters.synth.registry import get_template as dyn_get_template
        # OS family coarse-grained
        import platform
        fam = {"Linux":"linux","Darwin":"mac"}.get(platform.system(), "any")
        t = dyn_get_template(kind, fam)
        if t: return t
    except Exception:
        pass
    # 2) built-in mappings
    try:
        from adapters.mappings import CLI_TEMPLATES as BUILTIN
        tm = BUILTIN.get(kind)
        if tm:
            return tm.get(fam, tm.get("any"))
    except Exception:
        pass
    return None

# --- ניקוי/שיטוח פקודה למערך argv ---
def _render_cmd(template: str, params: Dict[str,Any]) -> List[str]:
    try:
        cmd_str = template.format(**params)
    except KeyError as e:
        raise ValidationFailed(f"missing_param:{e.args[0]}")
    # איסור פקודות מסוכנות
    forbidden = [r"\brm\s+-rf\b", r":\(\)\s*\{", r"\bmkfs\b", r"\bdd\s+if="]
    for patt in forbidden:
        if re.search(patt, cmd_str):
            raise ValidationFailed("blocked_by_policy")
    argv = shlex.split(cmd_str)
    if not argv:
        raise ValidationFailed("empty_command")
    return argv

# --- עטיפה עיקרית ---
async def run_adapter_with_assurance(user_id: str, kind: str, params: Dict[str,Any], execute: bool) -> Dict[str,Any]:
    # Consent חובה
    um = UserModel(UserStore("./assurance_store_users"))
    if not um.has_consent(user_id, "adapters/run"):
        # במערכת שלך: תבקש consent ותשמור; כאן מחזירים resource_required שקוף
        raise ResourceRequired("consent:adapters/run", "call user_model.consent_grant(uid,'adapters/run', ttl_seconds)")

    tmpl = _resolve_template(kind)
    if not tmpl:
        raise RefusedNotGrounded(f"unknown_adapter_kind:{kind}")

    # Kernel + Session
    kernel = AssuranceKernel("./assurance_store_adapters")
    sess = kernel.begin(f"adapter.{kind}", "1.0.0", f"Run adapter {kind}")
    # Claims: נדרוש שהפקודה לא ריקה, ואם execute – שה־rc==0
    sess.add_claim("cmd_non_empty", True)
    if execute:
        sess.add_claim("exit_code", 0)

    # Evidence: התבנית עצמה (חתימה על מקור)
    digest_tmpl = sess.cas.put_bytes(tmpl.encode("utf-8"), meta={"kind": kind, "type":"cli_template"})
    sess.add_evidence("template", f"adapters:{kind}", digest=digest_tmpl, trust=0.7)

    # Validators:
    sess.attach_validator(schema_validator({"required": [], "properties": {}}))  # מחייב הרצה של שאר הבדיקות בהמשך

    # Builder: DRY-RUN/EXECUTE דרך Executor
    argv = _render_cmd(tmpl, params)
    ex = SandboxExecutor("./executor/policy.yaml", "./adapters_sbx")

    async def _builder_dr():
        # DRY-RUN — לא מריצים; מאחסנים את argv והסבר
        payload = {"dry_run": True, "argv": argv}
        return [sess.cas.put_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), meta={"type":"dry-run"})]

    async def _builder_exec():
        # הרצה עם policy; ה-policy עצמו מגדיר כללי NET/FS/כלים
        try:
            rc, out = await ex.run(argv, inputs={}, allow_write=["out"], limits=Limits(
                cpu_seconds=ex.policy.cpu_seconds,
                mem_bytes=ex.policy.mem_bytes,
                wall_seconds=ex.policy.wall_seconds,
                open_files=ex.policy.open_files,
                no_net=ex.policy.no_net_default
            ))
        except ResourceRequired as e:
            raise
        payload = {"dry_run": False, "argv": argv, "exit_code": rc, "stdout_sha256": __import__("hashlib").sha256(out).hexdigest()}
        # evidence של stdout
        digest_stdout = sess.cas.put_bytes(out, meta={"channel":"stdout"})
        sess.add_evidence("stdout", "sandbox", digest=digest_stdout, trust=0.8)
        return [sess.cas.put_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), meta={"type":"exec-result"})]

    if execute:
        sess.set_builder(lambda s: asyncio.get_event_loop().run_until_complete(_builder_exec()))
    else:
        sess.set_builder(lambda s: asyncio.get_event_loop().run_until_complete(_builder_dr()))

    # Build + Validate + Commit
    outs = sess.build()
    # ולידציה פשוטה: יש ארטיפקט; אם execute – נוודא exit_code==0
    if execute:
        art = json.loads(kernel.cas.get_bytes(outs[0]).decode("utf-8"))
        if art.get("exit_code", 1) != 0:
            raise ValidationFailed("exit_code!=0")
    signer = Signer("adapters-hmac")
    rec = sess.commit(signer)
    return {"ok": True, "manifest_root": rec["root"], "artifact_digests": outs}
