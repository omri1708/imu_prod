# engine/pipeline_bindings.py
from typing import Dict, Any
from engine.adapters import register_all
from engine.registry import get
from contracts.base import AdapterResult

register_all()

def run_adapter(name: str, **kwargs) -> AdapterResult:
    fn = get(name)
    return fn(**kwargs)