# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from typing import Dict, Any

def record_resource_required(path: str, what: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "what": what}, ensure_ascii=False) + "\n")
