# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from assurance.errors import ResourceRequired

@dataclass
class CapabilitySpec:
    kind: str           # e.g. "tool.git" / "tool.curl"
    cmd: str            # executable name expected
    args_template: str  # e.g. "clone {repo} {dest}"
    schema: Dict[str, Any]
    install_hint: str   # how to obtain

class CapabilitySynthesizer:
    """
    When a ResourceRequired(tool:xxx) occurs, generate an adapter under adapters/generated/<kind>/
    with contract + cli_templates + pytest to integrate into the system.
    """
    def __init__(self, root: str = "./adapters/generated"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def generate(self, spec: CapabilitySpec) -> Dict[str, str]:
        slug = spec.kind.replace(".", "-")
        base = self.root / slug
        base.mkdir(parents=True, exist_ok=True)
        # contract.json
        (base / "contract.json").write_text(
            __import__("json").dumps(spec.schema, ensure_ascii=False, indent=2), encoding="utf-8")
        # cli_templates.json
        (base / "cli_templates.json").write_text(
            __import__("json").dumps({"kind": spec.kind, "templates": {"any": f"{spec.cmd} {spec.args_template}"}}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        # README
        (base / "README.md").write_text(
            f"# {spec.kind}\n\nInstall hint: {spec.install_hint}\n", encoding="utf-8")
        # pytest
        testf = Path("tests") / f"test_generated_{slug}.py"
        testf.write_text(f"""from fastapi.testclient import TestClient
from server.http_api import APP
c = TestClient(APP)
def test_{slug}_dryrun():
    r = c.post("/adapters/dry_run", json={{"user_id":"demo-user","kind":"{spec.kind}","params":{{}}}})
    assert r.status_code == 200
""", encoding="utf-8")
        return {"dir": str(base), "test": str(testf)}

    def from_exception(self, e: ResourceRequired) -> Dict[str,str] | None:
        # best-effort: parse "tool:xxx"
        if not e.what.startswith("tool:"):
            return None
        tool = e.what.split(":",1)[1]
        spec = CapabilitySpec(
            kind=f"tool.{tool}",
            cmd=tool,
            args_template="",
            schema={"type":"object", "properties":{}, "additionalProperties": True},
            install_hint=e.how_to_get or f"install '{tool}' via system package manager")
        return self.generate(spec)
