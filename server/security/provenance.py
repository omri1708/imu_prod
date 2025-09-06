# server/security/provenance.py
from typing import Dict, Any
from pathlib import Path
import json
import hashlib
import time

class ProvenanceStore:
    def __init__(self, base_dir: Path):
        self.base = base_dir
        self.base.mkdir(parents=True, exist_ok=True)

    def _write(self, kind: str, obj: Dict[str, Any]) -> str:
        payload = json.dumps(obj, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        p = self.base / f"{int(time.time())}_{kind}_{sha}.json"
        p.write_bytes(payload)
        return sha

    def record_capability(self, name: str, ok: bool, meta: Dict[str, Any]) -> str:
        return self._write("capability", {"name": name, "ok": ok, "meta": meta})

    def record_adapter_plan(self, adapter: str, plan: Dict[str, Any], dry: bool) -> str:
        return self._write("adapter_plan", {"adapter": adapter, "dry": dry, "plan": plan})

    def record_adapter_run(self, adapter: str, result: Dict[str, Any]) -> str:
        return self._write("adapter_run", {"adapter": adapter, "result": result})