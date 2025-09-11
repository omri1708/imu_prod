# engine/registry.py
from typing import Dict, Callable

_REGISTRY: Dict[str, Callable] = {}

def register(name: str, fn: Callable):
    _REGISTRY[name] = fn

def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"adapter_not_found:{name}")
    return _REGISTRY[name]

def list_adapters():
    return sorted(_REGISTRY.keys())