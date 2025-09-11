# PATH: engine/blueprints/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any
from importlib import import_module
from .generic_backend import generate_backend as generate_api
from .iac_terraform import generate as generate_iac_terraform
from .ci_github_actions import generate as generate_ci_github_actions
from .market_scan import generate as generate_market_scan
from .auto_writer import synthesize_blueprint

_REGISTRY: Dict[str, Callable[[Dict[str,Any]], Dict[str,bytes]]] = {
    "generic_backend": generate_api,
    "api": generate_api, 
    "custom": generate_api,
    "iac.terraform": generate_iac_terraform,
    "ci.github_actions": generate_ci_github_actions,
    "market.scan": generate_market_scan,
}

def _dynamic_load(name: str):
    # מנסה לייבא engine.blueprints.<name עם _ במקום .>
    mod_name = f"engine.blueprints.{name.replace('.','_')}"
    try:
        mod = import_module(mod_name)
    except Exception:
        # כותב קובץ מחולל חדש ואז מייבא
        path = synthesize_blueprint(name, {})
        mod = import_module(mod_name)
    fn = getattr(mod, "generate")
    _REGISTRY[name] = fn
    return fn

def resolve(name: str) -> Callable[[Dict[str,Any]], Dict[str,bytes]]:
    fn = _REGISTRY.get(name)
    if fn:
        return fn
    return _dynamic_load(name)  # ← עכשיו auto-writer מקבל עבודה כשמבקשים blueprint לא קיים


