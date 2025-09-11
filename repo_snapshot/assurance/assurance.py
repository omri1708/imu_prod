# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, shutil, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from .cas import CAS, sha256_bytes
from .errors import RefusedNotGrounded, ResourceRequired, ValidationFailed
from .signing import Signer
from .validators import Validator

def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

class DAGNode:
    def __init__(self, kind: str, params: Dict[str, Any] | None = None):
        self.kind = kind
        self.params = params or {}
        self.inputs: List[str] = []   # CAS digests
        self.outputs: List[str] = []  # CAS digests
        self.meta: Dict[str, Any] = {}
        self.ts = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "meta": self.meta,
            "ts": self.ts,
        }

class BuildSession:
    """
    One build session = one root artifact (Merkle root over nodes & content).
    """
    def __init__(self, kernel: "AssuranceKernel", op: str, version: str, description: str):
        self.kernel = kernel
        self.op = op
        self.version = version
        self.description = description
        self.cas = kernel.cas
        self.nodes: List[DAGNode] = []
        self.claims: List[Dict[str, Any]] = []
        self.evidence: List[Dict[str, Any]] = []
        self.validators: List[Validator] = []
        self.builder: Optional[Callable[["BuildSession"], List[str]]] = None
        self.context: Dict[str, Any] = {}   # free-form, used by validators
        self.outputs: List[str] = []
        self.audit: List[Dict[str, Any]] = []

    # ----- inputs, claims, evidence, validators -----
    def add_input_bytes(self, name: str, data: bytes, meta: Optional[Dict[str, Any]] = None) -> str:
        digest = self.cas.put_bytes(data, meta={"name": name, **(meta or {})})
        node = DAGNode("input", {"name": name})
        node.outputs = [digest]
        self.nodes.append(node)
        self.audit.append({"ts": time.time(), "event": "add_input_bytes", "name": name, "digest": digest})
        return digest

    def add_input_file(self, path: str, name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        digest = self.cas.put_file(path, meta={"name": name or os.path.basename(path), **(meta or {})})
        node = DAGNode("input", {"name": name or os.path.basename(path), "path": str(Path(path).resolve())})
        node.outputs = [digest]
        self.nodes.append(node)
        self.audit.append({"ts": time.time(), "event": "add_input_file", "name": name or os.path.basename(path), "digest": digest})
        return digest

    def add_claim(self, key: str, value: Any, units: Optional[str] = None, bounds: Optional[Dict[str,Any]] = None):
        c = {"key": key, "value": value, "units": units, "bounds": bounds}
        self.claims.append(c)
        self.audit.append({"ts": time.time(), "event": "add_claim", "claim": c})

    def add_evidence(self, kind: str, source: str, digest: Optional[str] = None,
                     trust: float = 0.5, timestamp: Optional[float] = None, signed: Optional[Dict[str,Any]] = None):
        ev = {"kind": kind, "source": source, "digest": digest, "trust": trust,
              "ts": timestamp or time.time(), "signature": signed}
        self.evidence.append(ev)
        self.audit.append({"ts": time.time(), "event": "add_evidence", "evidence": ev})

    def attach_validator(self, v: Validator):
        self.validators.append(v)
        self.audit.append({"ts": time.time(), "event": "attach_validator", "validator": v.name})

    def set_builder(self, builder: Callable[["BuildSession"], List[str]]):
        """Builder returns list of CAS digests for outputs; may raise ResourceRequired."""
        self.builder = builder

    # ----- build/validate/commit -----
    def build(self) -> List[str]:
        if not self.builder:
            raise RefusedNotGrounded("no builder set")
        outs = self.builder(self)
        if not outs or not all(isinstance(x, str) and len(x) == 64 for x in outs):
            raise ValidationFailed("builder did not return CAS digests")
        self.outputs = outs
        node = DAGNode("build", {"op": self.op, "version": self.version})
        # link latest input nodes as inputs:
        inputs = []
        for nd in self.nodes:
            if nd.kind == "input":
                inputs.extend(nd.outputs)
        node.inputs = inputs
        node.outputs = outs
        self.nodes.append(node)
        self.audit.append({"ts": time.time(), "event": "build", "outputs": outs})
        return outs

    def validate(self) -> List[Tuple[str, bool, str]]:
        ctx = {"claims": self.claims, "evidence": self.evidence, "artifact": self._artifact_payload(),
               "outputs": self.outputs, "op": self.op, "version": self.version}
        results = []
        ok_all = True
        for v in self.validators:
            ok, msg = v.run(ctx)
            results.append((v.name, ok, msg))
            ok_all = ok_all and ok
        if not ok_all:
            raise ValidationFailed("; ".join([m for (_,ok,m) in results if not ok]))
        self.audit.append({"ts": time.time(), "event": "validate", "results": results})
        return results

    def commit(self, signer: Signer) -> Dict[str, Any]:
        """Commit only if grounded: must have evidence + validators passed."""
        if not self.evidence:
            raise RefusedNotGrounded("no evidence provided")
        if not self.validators:
            raise RefusedNotGrounded("no validators attached")
        # ensure validate ran (or run now)
        self.validate()
        manifest = {
            "op": self.op,
            "version": self.version,
            "description": self.description,
            "claims": self.claims,
            "evidence": self.evidence,
            "nodes": [nd.to_dict() for nd in self.nodes],
            "outputs": self.outputs,
            "ts": time.time(),
        }
        root = sha256_bytes(canonical_json(manifest))
        sig = signer.sign(root.encode("utf-8"))
        record = {"root": root, "signature": sig, "manifest": manifest}
        # store manifest in CAS & audit log
        m_digest = self.cas.put_bytes(canonical_json(record), meta={"type": "manifest", "op": self.op})
        self.audit.append({"ts": time.time(), "event": "commit", "root": root, "manifest_digest": m_digest})
        # write audit log
        (self.kernel.audit_dir).mkdir(parents=True, exist_ok=True)
        with open(self.kernel.audit_dir / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "root": root, "op": self.op, "version": self.version,
                                "manifest_digest": m_digest, "events": self.audit}, ensure_ascii=False) + "\n")
        return {"root": root, "manifest_digest": m_digest, "signature": sig}

    def _artifact_payload(self) -> Dict[str, Any]:
        """Optionally materialize outputs for validators (keeping it light)."""
        # here we simply return a small dict linking outputs; validators can choose to fetch bytes from CAS.
        return {"output_digests": self.outputs}

class AssuranceKernel:
    def __init__(self, store_root: str = "./assurance_store"):
        self.root = Path(store_root).resolve()
        self.cas = CAS(str(self.root))
        self.audit_dir = self.root / "audit"

    def begin(self, op: str, version: str = "1.0.0", description: str = "") -> BuildSession:
        return BuildSession(self, op, version, description)
