# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, json
from assurance.assurance import AssuranceKernel
from assurance.validators import schema_validator
from assurance.signing import Signer
from assurance.errors import ResourceRequired, RefusedNotGrounded, ValidationFailed
from executor.sandbox import SandboxExecutor, Limits

async def run_command_with_assurance(cmd: list[str]) -> dict:
    kernel = AssuranceKernel("./assurance_store_bridge")
    sess = kernel.begin("run.command", "1.0.0", f"Run: {' '.join(cmd)}")

    # Claim: command should finish with rc==0 and produce stdout (<=1MB)
    sess.add_claim("exit_code", 0)
    sess.attach_validator(schema_validator({"required":[], "properties":{}}))

    ex = SandboxExecutor("./bridge_sandboxes")
    try:
        rc, out = await ex.run(cmd, inputs={}, allow_write=["out"], limits=Limits(no_net=True))
    except ResourceRequired as e:
        return {"ok": False, "resource_required": str(e)}
    # Pack artifact for validator (could add size checks etc.)
    payload = {"exit_code": rc, "stdout_sha256": __import__("hashlib").sha256(out).hexdigest()}
    sess.set_builder(lambda s: [s.cas.put_bytes(json.dumps(payload).encode("utf-8"), meta={"type":"run-result"})])

    sess.build()
    # Evidence: record stdout as evidence (CAS)
    sess.add_evidence("stdout", "sandbox:stdout", digest=sess.cas.put_bytes(out, meta={"channel":"stdout"}), trust=0.8)

    signer = Signer("bridge-hmac")
    rec = sess.commit(signer)
    return {"ok": True, "root": rec["root"], "manifest": rec["manifest_digest"]}
