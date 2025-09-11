# synth/specs_adapter.py
# -*- coding: utf-8 -*-
import yaml
import json
from typing import Dict, Any, List


def parse_adapter_jobs(spec_text: str) -> List[Dict[str, Any]]:
    spec = json.loads(spec_text)
    jobs = spec.get("adapters", []) or []
    out = []
    for j in jobs:
        kind = j.get("kind")
        if not kind: 
            continue
        out.append(dict(j))
    return out