# adapters/synth/generator.py
# Takes a declarative spec → generates adapter files under adapters/generated/<slug>/...
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import json, re, time, textwrap

GEN_ROOT = Path("adapters/generated")

@dataclass
class SynthSpec:
    name: str                  # human name
    kind: str                  # e.g. "db.migrate" or "unity.webgl"
    version: str
    description: str
    params: Dict[str, Any]     # schema-ish dict: {param: {type, required, enum?, pattern?, default?}}
    os_templates: Dict[str, str]  # {"linux|mac|win|any": "cmd with {param} format strings"}
    examples: Dict[str, Any] = None     # example params object
    capabilities: List[str] = None      # optional capabilities hints

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "-", s.lower()).strip("-")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _schema_from_params(params: Dict[str,Any]) -> Dict[str,Any]:
    required=[]
    props={}
    for k,info in params.items():
        typ=info.get("type","string")
        p={"type": typ}
        if "pattern" in info: p["pattern"]=info["pattern"]
        if "enum" in info: p["enum"]=info["enum"]
        if "minLength" in info: p["minLength"]=info["minLength"]
        if "default" in info: p["default"]=info["default"]
        props[k]=p
        if info.get("required", False): required.append(k)
    return {
        "$schema":"http://json-schema.org/draft-07/schema#",
        "title": "SynthAdapter",
        "type": "object",
        "required": required,
        "properties": props,
        "additionalProperties": False
    }

def create_adapter(spec: SynthSpec) -> Dict[str,Any]:
    slug = _slug(spec.kind)
    base = GEN_ROOT / slug
    _ensure_dir(base)

    # 1) contract.json
    contract = _schema_from_params(spec.params)
    (base / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) cli_templates.json
    cli = {"kind": spec.kind, "templates": spec.os_templates}
    (base / "cli_templates.json").write_text(json.dumps(cli, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) README.md
    readme = f"""# {spec.name} ({spec.kind})

    Version: {spec.version}

    {spec.description}

    ## Params Schema (excerpt)
    ```json
    {json.dumps(contract, ensure_ascii=False, indent=2)}
    Templates
    {json.dumps(cli, ensure_ascii=False, indent=2)}
    """
    (base / "README.md").write_text(readme, encoding="utf-8")
        
    # 4) test file under tests/
    test_code = textwrap.dedent(f"""
    # tests/test_generated_{slug}.py
    from fastapi.testclient import TestClient
    from server.http_api import APP

    client = TestClient(APP)

    def test_{slug}_dry_run_template():
        body = {{
            "user_id":"demo-user",
            "kind":"{spec.kind}",
            "params": {json.dumps(spec.examples or {}, ensure_ascii=False)}
        }}
        r = client.post("/adapters/dry_run", json=body)
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["ok"] is True
        assert "cmd" in j and j["cmd"]
        # assure placeholders were substituted
        for k in {json.dumps(list((spec.examples or {}).keys()))}:
                assert str({spec.examples or {}}[k]) in j["cmd"]
        """).strip()+"\n"
    tf = Path("tests") / f"test_generated_{slug}.py"
    tf.write_text(test_code, encoding="utf-8")

    meta = {
            "ok": True,
            "path": str(base),
            "kind": spec.kind,
            "files": ["contract.json","cli_templates.json","README.md", f"tests/test_generated_{slug}.py"]
        }
    return meta