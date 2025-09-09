from __future__ import annotations
import os, json
from typing import Dict, Any, List

SPEC_DIR = os.getenv("SPEC_DIR", "services/api/spec")

def _load(name: str):
    p = os.path.join(SPEC_DIR, name)
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else None

def load_entities_spec() -> List[Dict[str, Any]]:
    return _load("entities.json") or [{"name": "Item", "fields": [["name","str"],["value","float"]]}]

def load_behavior_spec() -> Dict[str, Any]:
    return _load("behavior.json") or {"name":"score", "inputs":["value"], "weights":[1.0], "tests":[{"inputs":{"value":2.5},"expected":2.5}]}
