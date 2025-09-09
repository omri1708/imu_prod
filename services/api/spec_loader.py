# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Spec loader:
- Tries to load JSON specs written by the blueprint:
  services/api/spec/entities.json
  services/api/spec/behavior.json
- Provides safe defaults if files are missing.
"""

import json, os
from typing import Dict, Any, List

SPEC_DIR = os.getenv("SPEC_DIR", "services/api/spec")

def _load_json(fname: str):
    path = os.path.join(SPEC_DIR, fname)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_entities_spec() -> List[Dict[str, Any]]:
    data = _load_json("entities.json")
    if isinstance(data, list):
        return data
    # fallback
    return [
        {"name": "Item", "fields": [["name", "str"], ["value", "float"]]}
    ]

def load_behavior_spec() -> Dict[str, Any]:
    data = _load_json("behavior.json")
    if isinstance(data, dict):
        return data
    # fallback example
    return {
        "name": "score",
        "inputs": ["value"],
        "weights": [1.0],
        "tests": [{"inputs": {"value": 2.5}, "expected": 2.5}]
    }
