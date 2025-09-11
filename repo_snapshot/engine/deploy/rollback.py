# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from engine.artifacts.registry import latest

def suggest_rollback(name: str) -> Dict[str, Any]:
    info = latest(f"build-{name}")
    if not info: return {"available": False}
    return {"available": True, "to_digest": info.get("digest"), "files": info.get("files", [])}
